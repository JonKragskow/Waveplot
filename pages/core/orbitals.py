import numpy as np


def radial_s(n, rho):
    """
    Calculates radial Wavefunction of s orbital
    for the specified principal quantum number

    Parameters
    ----------
    n : int
        principal quantum number
    rho : np.ndarray
        values of rho = 2.*r/n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    np.ndarray
        radial wavefunction as a function of rho
    """

    if n == 1:
        rad = 2.*np.exp(-rho/2.)
    if n == 2:
        rad = 1./(2.*np.sqrt(2.))*(2.-rho)*np.exp(-rho/2.)
    if n == 3:
        rad = 1./(9.*np.sqrt(3.))*(6.-6.*rho+rho**2.)*np.exp(-rho/2.)
    if n == 4:
        rad = (1./96.)*(24.-36.*rho+12.*rho**2.-rho**3.)*np.exp(-rho/2.)
    if n == 5:
        rad = (1./(300.*np.sqrt(5.)))*(120.-240.*rho+120.*rho**2.-20.*rho**3.+rho**4.)*np.exp(-rho/2.) # noqa
    if n == 6:
        rad = (1./(2160.*np.sqrt(6.)))*(720.-1800.*rho+1200.*rho**2.-300.*rho**3.+30.*rho**4.-rho**5.)*np.exp(-rho/2.) # noqa
    if n == 7:
        rad = (1./(17640.*np.sqrt(7)))*(5040. - 15120.*rho + 12600.*rho**2. - 4200.*rho**3. + 630.*rho**4. -42* rho**5. + rho**6)*np.exp(-rho/2.) # noqa

    return rad


def radial_p(n, rho):
    """
    Calculates radial Wavefunction of p orbital
    for the specified principal quantum number

    Parameters
    ----------
    n : int
        principal quantum number
    rho : np.ndarray
        values of rho = 2.*r/n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    np.ndarray
        radial wavefunction as a function of rho
    """

    if n == 2:
        rad = 1./(2.*np.sqrt(6.))*rho*np.exp(-rho/2.)
    elif n == 3:
        rad = 1./(9.*np.sqrt(6.))*rho*(4.-rho)*np.exp(-rho/2.)
    elif n == 4:
        rad = 1./(32.*np.sqrt(15.))*rho*(20.-10.*rho+rho**2.)*np.exp(-rho/2.)
    elif n == 5:
        rad = 1./(150.*np.sqrt(30.))*rho*(120.-90.*rho+18.*rho**2.-rho**3.)*np.exp(-rho/2.) # noqa
    elif n == 6:
        rad = 1./(432.*np.sqrt(210.))*rho*(840.-840.*rho+252.*rho**2.-28.*rho**3.+rho**4.)*np.exp(-rho/2.) # noqa
    elif n == 7:
        rad = 1./(11760.*np.sqrt(21.))*rho*(6720. - 8400.*rho+3360.*rho**2.-560.*rho**3.+40*rho**4. - rho**5)*np.exp(-rho/2.) # noqa
    return rad


def radial_d(n, rho):
    """
    Calculates radial Wavefunction of d orbital
    for the specified principal quantum number

    Parameters
    ----------
    n : int
        principal quantum number
    rho : np.ndarray
        values of rho = 2.*r/n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    np.ndarray
        radial wavefunction as a function of rho
    """

    if n == 3:
        rad = 1./(9.*np.sqrt(30.))*rho**2.*np.exp(-rho/2.)
    elif n == 4:
        rad = 1./(96.*np.sqrt(5.))*(6.-rho)*rho**2.*np.exp(-rho/2.)
    elif n == 5:
        rad = 1./(150.*np.sqrt(70.))*(42.-14.*rho+rho**2)*rho**2.*np.exp(-rho/2.) # noqa
    elif n == 6:
        rad = 1./(864.*np.sqrt(105.))*(336.-168.*rho+24.*rho**2.-rho**3.)*rho**2.*np.exp(-rho/2.) # noqa
    elif n == 7:
        rad = 1./(7056.*np.sqrt(105.))*(3024. - 2016.*rho + 432.*rho**2. -36* rho**3. + rho**4)*rho**2.*np.exp(-rho/2.) # noqa

    return rad


def radial_f(n, rho):
    """
    Calculates radial wavefunction of f orbital
    for the specified principal quantum number

    Parameters
    ----------
    n : int
        principal quantum number
    rho : np.ndarray
        values of rho = 2.*r/n, where r^2 = x^2+y^2+z^2

    Returns
    -------
    np.ndarray
        radial wavefunction as a function of rho
    """

    if n == 4:
        rad = 1./(96.*np.sqrt(35.))*rho**3.*np.exp(-rho/2.)
    elif n == 5:
        rad = 1./(300.*np.sqrt(70.))*(8.-rho)*rho**3.*np.exp(-rho/2.)
    elif n == 6:
        rad = 1./(2592.*np.sqrt(35.))*(rho**2.-18.*rho+72.)*rho**3.*np.exp(-rho/2.) # noqa
    elif n == 7:
        rad = 1./(17640.*np.sqrt(42.))*(-rho**3 + 30*rho**2. - 270.*rho + 720.)*rho**3.*np.exp(-rho/2.) # noqa

    return rad


def s_3d(n, cutaway=1.):
    """
    Calculates s orbital wavefunction on a grid

    Parameters
    ----------
        n : int
            prinipal quantum number of orbital
        cutaway : int
            number used to split orbital in half
    Returns
    -------
        x : np.mgrid
            x values
        y : np.mgrid
            y values
        z : np.mgrid
            z values
        wav : np.mgrid
            wavefunction values at x, y, z
        upper : float
            max value of axes
        lower : float
            min value of axes
        ival : float
            isoval for orbital plotting
    """

    if n == 1:
        upper = 10.
        step = 2*upper/50.
        lower = - upper
    elif n == 2:
        upper = 17.
        step = 2*upper/50.
        lower = - upper
    elif n == 3:
        upper = 30.
        step = 2*upper/50.
        lower = - upper
    elif n == 4:
        upper = 45.
        step = 2*upper/50.
        lower = - upper
    elif n == 5:
        upper = 58.
        step = 2*upper/60.
        lower = - upper
    elif n == 6:
        upper = 75.
        step = 2*upper/70.
        lower = - upper

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_s(n, 2*r/n)

    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.5/np.sqrt(np.pi)

    wav = ang*rad

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, 0.0005, step


def p_3d(n, cutaway=1.):
    """
    Calculates p orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

    if n == 1:
        upper = 10.
        step = 2*upper/50.
        lower = - upper
    elif n == 2:
        upper = 17.
        step = 2*upper/50.
        lower = - upper
    elif n == 3:
        upper = 30.
        step = 2*upper/50.
        lower = - upper
    elif n == 4:
        upper = 45.
        step = 2*upper/50.
        lower = - upper
    elif n == 5:
        upper = 60.
        step = 2*upper/60.
        lower = - upper
    elif n == 6:
        upper = 80.
        step = 2*upper/70.
        lower = - upper

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_p(n, 2*r/n)

    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = np.sqrt(3./(4.*np.pi)) * z/r

    wav = ang*rad

    if n == 1:
        ival = 0.0005
    if n == 2:
        ival = 0.0005
    if n == 3:
        ival = 0.0005
    elif n == 4:
        ival = 0.0005
    elif n == 5:
        ival = 0.0005
    elif n == 6:
        ival = 0.0005

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, ival, step


def dz_3d(n, cutaway=1.):
    """
    Calculates dz2 orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

    if n == 3:
        upper = 50.
        step = 2*upper/60.
        lower = - upper
    elif n == 4:
        upper = 70.
        step = 2*upper/70.
        lower = - upper
    elif n == 5:
        upper = 98.
        step = 2*upper/80.
        lower = - upper
    elif n == 6:
        upper = 135.
        step = 2*upper/90.
        lower = - upper

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_d(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 2*z**2-x**2-y**2

    wav = rad*ang

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, 0.08, step


def dxy_3d(n, cutaway=1.):
    """
    Calculates dxy orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

    if n == 3:
        upper = 45.
        step = 2*upper/60.
        lower = - upper
    elif n == 4:
        upper = 70.
        step = 2*upper/70.
        lower = - upper
    elif n == 5:
        upper = 98.
        step = 2*upper/80.
        lower = - upper
    elif n == 6:
        upper = 135.
        step = 2*upper/90.
        lower = - upper

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_d(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = x*y

    wav = rad*ang

    if n == 3:
        ival = 0.005
    elif n == 4:
        ival = 0.01
    elif n == 5:
        ival = 0.01
    elif n == 6:
        ival = 0.01

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, ival, step


def fz_3d(n, cutaway=1.):
    """
    Calculates fz3 orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

    if n == 4:
        upper = 70.
        step = 2*upper/60.
        lower = - upper
    elif n == 5:
        upper = 100.
        step = 2*upper/75.
        lower = - upper
    elif n == 6:
        upper = 130.
        step = 2*upper/85.
        lower = - upper

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_f(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.25 * np.sqrt(7/np.pi) * z*(2*z**2-3*x**2-3*y**2)/(r**3)

    wav = rad*ang

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, 0.000005, step


def fxyz_3d(n, cutaway=1.):
    """
    Calculates fxyz orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

    if n == 4:
        upper = 60.
        step = 2*upper/60.
        lower = - upper
    elif n == 5:
        upper = 90.
        step = 2*upper/70.
        lower = - upper
    elif n == 6:
        upper = 115.
        step = 2*upper/80.
        lower = - upper

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_f(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.5 * np.sqrt(105/np.pi) * x*y*z/(r**3)

    wav = rad*ang

    if n == 4:
        ival = 0.000005
    elif n == 5:
        ival = 0.000005
    elif n == 6:
        ival = 0.000005

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, ival, step


def fyz2_3d(n, cutaway=1.):
    """
    Calculates fyz2 orbital wavefunction on a grid

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    cutaway : int
        number used to split orbital in half
    Returns
    -------
    x : np.mgrid
        x values
    y : np.mgrid
        y values
    z : np.mgrid
        z values
    wav : np.mgrid
        wavefunction values at x, y, z
    upper : float
        max value of axes
    lower : float
        min value of axes
    ival : float
        isoval for orbital plotting
    """

    if n == 4:
        upper = 65.
        step = 2*upper/60.
        lower = - upper
    elif n == 5:
        upper = 90.
        step = 2*upper/90.
        lower = - upper
    elif n == 6:
        upper = 125.
        step = 2*upper/100.
        lower = - upper

    x, y, z = np.meshgrid(
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step),
        np.arange(lower, upper + step, step)
    )

    r = np.sqrt(x**2 + y**2 + z**2)

    rad = radial_f(n, 2*r/n)
    rad[np.where(y > lower + (upper-lower)*cutaway)] = 0.

    ang = 0.25 * np.sqrt(35/(2*np.pi)) * (3*x**2-y**2)*y/r**3

    wav = rad*ang

    n_points = np.shape(x)[0]

    return n_points, wav, upper, lower, 0.000005, step


def s_2d(n, r, wf_type):
    """
    Calculates s orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    r : np.ndarray
        values of distance r
    wf_type : string {'RDF', 'RWF'}
        type of wavefunction to calculate
    Returns
    -------
    np.ndarray
        y values corresponding to radial wavefunction or radial
        distribution function
    """

    if "RDF" in wf_type:
        return r**2. * radial_s(n, 2.*r/n)**2
    if "RWF" in wf_type:
        return radial_s(n, 2.*r/n)


def p_2d(n, r, wf_type):
    """
    Calculates p orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    r : np.ndarray
        values of distance r
    wf_type : string {'RDF', 'RWF'}
        type of wavefunction to calculate
    Returns
    -------
    np.ndarray
        y values corresponding to radial wavefunction or radial
        distribution function
    """

    if "RDF" in wf_type:
        return r**2. * radial_p(n, 2.*r/n)**2
    elif "RWF" in wf_type:
        return radial_p(n, 2.*r/n)


def d_2d(n, r, wf_type):
    """
    Calculates d orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    r : np.ndarray
        values of distance r
    wf_type : string {'RDF', 'RWF'}
        type of wavefunction to calculate

    Returns
    -------
    np.ndarray
        y values corresponding to radial wavefunction or radial
        distribution function
    """

    if "RDF" in wf_type:
        return r**2. * radial_d(n, 2.*r/n)**2
    if "RWF" in wf_type:
        return radial_d(n, 2.*r/n)


def f_2d(n, r, wf_type):
    """
    Calculates f orbital radial wavefunction or radial distribution function

    Parameters
    ----------
    n : int
        prinipal quantum number of orbital
    r : np.ndarray
        values of distance r
    wf_type : string {'RDF', 'RWF'}
        type of wavefunction to calculate

    Returns
    -------
    np.ndarray
        y values corresponding to radial wavefunction or radial
        distribution function
    """

    if "RDF" in wf_type:
        return r**2. * radial_f(n, 2.*r/n)**2
    if "RWF" in wf_type:
        return radial_f(n, 2.*r/n)
