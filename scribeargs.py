from scribe import ascii_alphabet

scribe_args = {
    # 'alphabet': 'hindu_alphabet',
    'alphabet': ascii_alphabet,

    'noise': .05,

    'vbuffer': 3,
    'hbuffer': 5,

    # 'avg_seq_len': 0,          # Length is adjusted & varied according to nchars_per_sample
    # 'varying_len': False,
    # 'nchars_per_sample': 3,

    'avg_seq_len': 120,
    'varying_len': False,
    'nchars_per_sample': None,     # Varies to fit the length

    # 'avg_seq_len': 50,            # Length varies +/- 25%
    # 'varying_len': True,
    # 'nchars_per_sample': None,
}
