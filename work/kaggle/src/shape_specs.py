"""Same dense architecture/FFN rule as L1; shapes alone change."""
SHAPES = [(6,6,512), (6,8,640), (6,6,768), (6,6,640), (12,6,640),
          (10,4,768), (8,6,640), (6,10,512), (4,4,768),
          (8,8,640), (8,4,768), (6,5,768), (7,7,640)]


def name(shape):
    e, d, w = shape
    return f'e{e}-d{d}-w{w}'


def spec(shape):
    e, d, w = shape
    # C=1368, D=1712: ceil(8*d/3) to a multiple of 8.
    return dict(d_model=w, heads=w//64, encoder_layers=e, decoder_layers=d,
                swiglu_dim=((8*w+23)//24)*8, vocab=8195, max_positions=2048, dropout=0.)


def parameter_formula(shape):
    e, d, w = shape; f = spec(shape)['swiglu_dim']
    encoder = e*(4*w*w+3*w*f+9*w+2*f)+2*w
    decoder = d*(8*w*w+3*w*f+15*w+2*f)+2*w
    embeddings = (8195+2048)*w
    head = 8195*(w+1)
    return dict(encoder=encoder, decoder=decoder, embeddings=embeddings, head=head,
                total=encoder+decoder+embeddings+head)
