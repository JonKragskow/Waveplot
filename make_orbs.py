from pages.core.orbitals import calc_wav

orbs = [
    '1s',
    '2s',
    '3s',
    '4s',
    '5s',
    '6s',
    '2p',
    '3p',
    '4p',
    '5p',
    '6p',
    '3dz2',
    '4dz2',
    '5dz2',
    '6dz2',
    '3dxy',
    '4dxy',
    '5dxy',
    '6dxy',
    '4fz3',
    '5fz3',
    '6fz3',
    '4fxyz',
    '5fxyz',
    '6fxyz',
    '4fyz2',
    '5fyz2',
    '6fyz2',
    'sp',
    'sp2',
    'sp3'
]

for orb in orbs:
    calc_wav(orb)
    calc_wav(orb, 'x')
    calc_wav(orb, 'y')
    calc_wav(orb, 'z')
