"""MoleculeACE 모델을 애플 실리콘 GPU(MPS)에서 돌리기 위한 호환 계층.

문제
----
MoleculeACE 의 모든 torch 모델이 장치를 이렇게 고른다.

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

맥에는 CUDA 가 없으므로 **무조건 CPU 로 떨어진다.** MPS 는 쓸 수 있는데
코드가 볼 생각을 하지 않는다. 이 줄은 gcn·gat·mpnn·afp·cnn·mlp·transformer
일곱 파일에 각각 박혀 있어서 한 군데를 고쳐서 될 일이 아니다.

왜 굳이 MPS 를 쓰나 — ⚠️ 모델마다 답이 다르다
---------------------------------------------
2026-07-29 실측(M5 Pro 18코어. CPU 쪽은 다른 샤드 7개와 경합하는 실제 조건):

    MPNN  5에폭 · CHEMBL204_Ki(2,201분자)   CPU 341.3초  →  MPS  54.9초  (⭐ 6.2배 빠름)
    GCN   3에폭 · CHEMBL2835_Ki(가장 작음)  CPU   5.5초  →  MPS  11.1초  (2배 느림)

**"GPU 가 빠르다/느리다" 를 한 줄로 말할 수 없다.** 분자 그래프는 작아서 커널
실행 오버헤드가 계산량을 넘기 쉽고, 그 손익분기가 모델 무게에 따라 갈린다.
가벼운 GCN 은 CPU 가 낫고, 엣지 네트워크를 도는 MPNN 은 GPU 가 압도적이다.
**모델마다 재보고 정해야 한다.**

구현 — 왜 몽키패치가 아니라 서브클래스인가
------------------------------------------
처음에는 `torch.device` 를 감싸서 "cuda" 요청을 "mps" 로 돌리려 했다. **깨진다.**
torch/PyG 안에 `torch.device | None` 같은 PEP 604 타입 표기가 있어서,
`torch.device` 가 함수가 되는 순간 `unsupported operand type(s) for |` 로 죽는다.

그래서 모델 클래스만 감싼다. `cross_validate` 가 내부에서
`f = model(**hyperparameters)` 로 직접 만들기 때문에 인스턴스를 넘길 수는 없고,
**클래스를 넘기는 것은 가능**하다.

⚠️ 서브클래스는 반드시 **모듈 최상위에 등록**해야 한다. 조기종료가 모델을
pickle 하는데 함수 지역 클래스는 pickle 되지 않는다(pyg_compat 의 CompatGMT 에서
이미 한 번 겪었다). 그래서 `globals()` 에 넣어 pickle 이 찾을 수 있게 한다.

optimizer 는 `cls.__init__` 안에서 CPU 파라미터로 만들어지지만, `Module.to()` 가
파라미터를 **제자리에서** 옮기므로 optimizer 가 들고 있는 참조는 그대로 유효하다.

⚠️ 수치가 달라진다
------------------
MPS 는 부동소수점 연산 순서와 커널 구현이 CPU 와 다르므로 **같은 시드라도 결과가
정확히 일치하지 않는다.** 이 경로로 낸 값을 기록할 때는 반드시 `device=mps` 를
함께 적어야 한다. (`run_deep.py` 는 결과 CSV 에 `device` 열을 남긴다.)

쓰는 법
------
    from device_compat import to_mps
    from MoleculeACE import MPNN
    cross_validate(to_mps(MPNN), data, ...)
"""
import os

# PyG 의 일부 연산은 MPS 커널이 없다. 없으면 CPU 로 되돌아가게 둔다
# (안 켜면 NotImplementedError 로 죽는다). torch 임포트 전에 켜야 확실하다.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_REG = {}


def mps_available() -> bool:
    import torch
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class _MPSMixin:
    """__init__ 이 끝난 뒤 장치를 MPS 로 바꾸고 모델을 옮긴다."""

    def __init__(self, *args, **kwargs):
        import torch
        super().__init__(*args, **kwargs)
        self.device = torch.device("mps")
        self.model = self.model.to(self.device)


def to_mps(cls):
    """cls 의 MPS 판 서브클래스를 돌려준다. MPS 가 없으면 cls 를 그대로 돌려준다.

    같은 클래스로 두 번 부르면 같은 서브클래스를 돌려준다(pickle 정체성 유지).
    """
    if not mps_available():
        return cls
    name = "MPS" + cls.__name__
    if name in _REG:
        return _REG[name]
    new = type(name, (_MPSMixin, cls), {})
    new.__module__ = __name__
    globals()[name] = new       # pickle 이 이름으로 찾을 수 있도록 모듈 전역에 등록
    _REG[name] = new
    return new
