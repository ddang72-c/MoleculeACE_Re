"""MoleculeACE 의 GNN 모델을 최신 PyTorch Geometric 에서 돌리기 위한 호환 계층.

문제
----
MoleculeACE(2022) 는 PyG 2.0.4 의 GraphMultisetTransformer 를 이렇게 부른다.

    GraphMultisetTransformer(in_channels=..., hidden_channels=..., out_channels=..., num_heads=8)
    forward(x, batch, edge_index)

PyG 2.5 부터 이 클래스가 다시 쓰이면서 시그니처가 바뀌었다.

    GraphMultisetTransformer(channels, k, num_encoder_blocks=1, heads=1, layer_norm=False, dropout=0.0)
    forward(x, index=None, ptr=None, dim_size=None, ...)

그래서 GCN·GAT·MPNN 이 임포트 시점에 죽는다.

    TypeError: GraphMultisetTransformer.__init__() got an unexpected keyword argument 'in_channels'

바뀜 점이 이름만이 아니다. 옛 버전은 in→hidden→out 으로 차원을 바꿔주는데
새 버전은 channels 를 그대로 돌려준다. 그래서 out_channels 로 보내는 선형
투영을 이쪽에서 따로 붙여야 원래 구조가 유지된다.

쓰는 법
------
MoleculeACE 의 모델을 임포트하기 **전에** 먼저 부른다.

    from pyg_compat import patch
    patch()
    from MoleculeACE import GCN, GAT, MPNN   # 이제 임포트된다

⚠️ 주의
------
이것은 원 구현과 **동일한 계산이 아니다**. 새 GMT 는 k 개의 대표 원소로
모으는 attention pooling 이고 옛 구현과 내부가 다르다. 따라서 이 경로로 난
GNN 수치를 공식 결과표와 셀 단위로 대조하면 안 된다. 평균 수준의 참고값으로만
쓰고, 인용할 때는 이 패치를 썼다는 사실을 함께 적어야 한다.

대안은 PyG 를 2.0.4 로 핀하는 것인데, 그러면 numpy 2 와 충돌해 rdkit 을
올릴 수 없게 되므로 이 저장소는 패치 쪽을 택했다.

CompatGMT 는 모듈 최상위에 둔다. MoleculeACE 가 학습 중 모델을 pickle 로
저장하는데, 함수 안에 정의된 클래스는 pickle 되지 않기 때문이다.
"""
import torch
from torch.nn import Linear

# patch() 가 채운다
_K = 8
_BLOCKS = 1


class CompatGMT(torch.nn.Module):
    """옛 in/hidden/out 인터페이스를 새 GraphMultisetTransformer 위에 얹는다."""

    def __init__(self, in_channels, hidden_channels=None, out_channels=None,
                 num_heads=1, **_ignored):
        super().__init__()
        from torch_geometric.nn import GraphMultisetTransformer as _NewGMT
        out_channels = out_channels or in_channels
        self.pool = _NewGMT(channels=in_channels, k=_K,
                            num_encoder_blocks=_BLOCKS, heads=num_heads)
        # 새 GMT 는 channels 를 그대로 돌려주므로 out_channels 로 투영한다
        self.proj = (Linear(in_channels, out_channels)
                     if out_channels != in_channels else torch.nn.Identity())

    def forward(self, x, batch=None, edge_index=None, **_ignored):
        # 옛 시그니처는 (x, batch, edge_index). 새 GMT 는 edge_index 를 안 쓴다.
        return self.proj(self.pool(x, index=batch))


def patch(k: int = 8, num_encoder_blocks: int = 1):
    """MoleculeACE 의 GNN 모듈이 쓰는 GraphMultisetTransformer 를 교체한다.

    :param k: 새 GMT 가 모을 대표 원소 개수 (attention pooling 의 seed 수)
    :param num_encoder_blocks: 대표 원소끼리의 self-attention 블록 수
    """
    global _K, _BLOCKS
    _K, _BLOCKS = k, num_encoder_blocks

    import MoleculeACE.models.gcn as _gcn
    import MoleculeACE.models.gat as _gat
    import MoleculeACE.models.mpnn as _mpnn
    for mod in (_gcn, _gat, _mpnn):
        mod.GraphMultisetTransformer = CompatGMT
    return CompatGMT
