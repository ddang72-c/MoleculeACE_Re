"""MoleculeACE 의 LSTM 을 Keras 3 에서 돌리기 위한 호환 계층.

문제 ① — fit() 시그니처가 바뀌었다
----------------------------------
MoleculeACE(2022) 는 `lstm.py` 에서 이렇게 부른다.

    self.model.fit(tr_generator, use_multiprocessing=True, epochs=epochs, workers=1, verbose=1)

Keras 3 는 `use_multiprocessing` 과 `workers` 를 fit() 에서 없앴다
(이제 tf.data 파이프라인 쪽에서 처리한다). 그래서 학습이 시작되자마자 죽는다.

    TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'use_multiprocessing'

문제 ② — 사전학습 가중치가 배포되지 않는다
------------------------------------------
`LSTM.__init__` 의 기본값은 이 경로를 가리킨다.

    MoleculeACE/Data/pretrained_models/pretrained_lstm.h5

**이 파일은 pip 패키지에도, GitHub 저장소에도 없다.** 저장소에는 그것을 만드는
`Experiments/pretrain_lstm.py` 만 있다. 파일이 없으면 `__init__` 은 경고만 내고
`self.model` 을 아예 만들지 않는다(else 가 없다). 그래서 다음 단계에서 죽는다.

    AttributeError: 'LSTM' object has no attribute 'model'

⚠️ 그래서 공식 LSTM 수치(RMSE 0.7423, 24접근 중 5위)는 **배포된 산출물만으로는
재현할 수 없다.** 두 가지 길이 있다.

  (a) `pretrained_model=None` — 사전학습 없이 처음부터 학습한다. 돌아가지만
      공식 수치와 같은 조건이 아니다. 공식 LSTM 은 next-token 사전학습
      가중치로 초기화한 전이학습이다. 셀 단위든 평균이든 대조하면 안 된다.
  (b) 사전학습을 직접 재현한다. `pretrain_lstm.py` 는 외부 코퍼스가 필요없고
      벤치마크 30표적의 train SMILES 를 10배 증강해 100에폭 돌린다.
      비용은 크지만 자기완결적이다.

이 모듈은 (a) 를 쓸 수 있게만 해준다. (b) 를 하면 이 경고는 없어진다.

쓰는 법
------
MoleculeACE 의 LSTM 을 **학습시키기 전에** 부른다.

    from keras_compat import patch
    patch()
    m = LSTM(pretrained_model=None, **hp)   # 사전학습 없음 — 위 ⚠️ 참고
"""
import warnings

_PATCHED = False


def patch():
    """Keras 3 가 없앤 fit() 인자를 조용히 걸러내도록 Sequential.fit 을 감싼다."""
    global _PATCHED
    if _PATCHED:
        return
    import keras

    _orig_fit = keras.Sequential.fit

    def _fit(self, *args, **kwargs):
        # Keras 3 에서 사라진 인자들. 값이 무엇이든 동작에 영향이 없다
        # (multiprocessing 은 이제 입력 파이프라인 쪽 책임이다).
        kwargs.pop("use_multiprocessing", None)
        kwargs.pop("workers", None)
        kwargs.pop("max_queue_size", None)
        return _orig_fit(self, *args, **kwargs)

    keras.Sequential.fit = _fit
    _PATCHED = True


def make_lstm(cls, hp, pretrained_path=None):
    """LSTM 을 안전하게 만든다.

    :param cls: MoleculeACE 의 LSTM 클래스
    :param hp: get_benchmark_config 가 준 하이퍼파라미터
    :param pretrained_path: 사전학습 .h5 경로. None 이면 처음부터 학습한다.
    """
    patch()
    if pretrained_path is None:
        warnings.warn(
            "사전학습 가중치 없이 LSTM 을 만든다. 공식 결과표의 LSTM(0.7423)은 "
            "전이학습 수치이므로 이 값과 대조하면 안 된다.")
    return cls(pretrained_model=pretrained_path, **hp)
