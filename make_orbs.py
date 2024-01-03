from pages.core.orbitals import calc_wav

for half in ['', 'x', 'y', 'z']:
    for n in range(1, 7):
        calc_wav(f'{n:d}s+0', half=half)
    for n in range(2, 7):
        for ml in range(-1, 2):
            calc_wav(f'{n:d}p{ml:+d}', half=half)
    for n in range(3, 7):
        for ml in range(-2, 3):
            calc_wav(f'{n:d}d{ml:+d}', half=half)
    for n in range(4, 7):
        for ml in range(-3, 4):
            calc_wav(f'{n:d}f{ml:+d}', half=half)
        calc_wav(f'{n:d}f+3c', half=half)
        calc_wav(f'{n:d}f-3c', half=half)
        calc_wav(f'{n:d}f+2c', half=half)
        calc_wav(f'{n:d}f-2c', half=half)
    calc_wav('sp', half=half)
    calc_wav('sp2', half=half)
    calc_wav('sp3', half=half)
