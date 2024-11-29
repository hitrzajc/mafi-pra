import numpy
def verlet( f, y0, t_eval, method,t_span ):
    x0, v0 = y0
    """Verlet's 2nd order symplectic method

    USAGE:
        (x,v) = varlet(f, x0, v0, t_eval)

    INPUT:
        f     - function of x and t_eval equal to d^2x/dt^2.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s) of x.  Specifies the value of x when
                t_eval = t_eval[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        v0    - the initial condition(s) of v=dx/dt.  Specifies the value of v when
                t_eval = t_eval[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t_eval     - list or NumPy array of t_eval values to compute solution at.
                t_eval[0] is the the initial condition point, and the difference
                h=t_eval[i+1]-t_eval[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values for x corresponding to each
                entry in t_eval array.  If a system is being solved, x will be
                an array of arrays.
        v     - NumPy array containing solution values for v=dx/dt corresponding to each
                entry in t_eval array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This function used the Varlet/Stoermer/Encke (symplectic) method
        method to solve the initial value problem

            dx^2
            -- = f(x),     x(t_eval(1)) = x0  v(t_eval(1)) = v0
            dt^2

        at the t_eval values stored in the t_eval array (so the interval of solution is
        [t_eval[0], t_eval[N-1]].  The 3rd-order Taylor is used to generate
        the first values of the solution.

    """
    n = len( t_eval )
    x = numpy.array( [ x0 ] * n )
    v = numpy.array( [ v0 ] * n )
    for i in range( n - 1 ):
        h = t_eval[i+1] - t_eval[i]
        x[i+1] = x[i] + h * v[i] + (h*h/2) * f(x[i])
        v[i+1] = v[i] + (h/2) * ( f(x[i])+f(x[i+1]))

    return numpy.array([x,v])

def pefrl( f,y0, t_eval, method , t_span):
    x0, v0 = y0
    """Position Extended Forest-Ruth Like 4th order symplectic method by Omelyan et al.

    USAGE:
        (x,v) = varlet(f, x0, v0, t_eval)

    INPUT:
        f     - function of x and t_eval equal to d^2x/dt^2.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s) of x.  Specifies the value of x when
                t_eval = t_eval[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        v0    - the initial condition(s) of v=dx/dt.  Specifies the value of v when
                t_eval = t_eval[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t_eval     - list or NumPy array of t_eval values to compute solution at.
                t_eval[0] is the the initial condition point, and the difference
                h=t_eval[i+1]-t_eval[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values for x corresponding to each
                entry in t_eval array.  If a system is being solved, x will be
                an array of arrays.
        v     - NumPy array containing solution values for v=dx/dt corresponding to each
                entry in t_eval array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This function uses the Omelyan et al (symplectic) method
        method to solve the initial value problem

            dx^2
            -- = f(x),     x(t_eval(1)) = x0  v(t_eval(1)) = v0
            dt^2

        at the t_eval values stored in the t_eval array (so the interval of solution is
        [t_eval[0], t_eval[N-1]].

    """

    xsi=0.1786178958448091
    lam=-0.2123418310626054
    chi=-0.6626458266981849e-1
    n = len( t_eval )
    x = numpy.array( [ x0 ] * n )
    v = numpy.array( [ v0 ] * n )
    for i in range( n - 1 ):
        h = t_eval[i+1] - t_eval[i]
        y=numpy.copy(x[i])
        w=numpy.copy(v[i])
        y += xsi*h*w
        w += (1-2*lam)*(h/2)*f(y)
        y += chi*h*w
        w += lam*h*f(y)
        y += (1-2*(chi+xsi))*h*w
        w += lam*h*f(y)
        y += chi*h*w
        w += (1-2*lam)*(h/2)*f(y)
        y += xsi*h*w
        x[i+1]=numpy.copy(y)
        v[i+1]=numpy.copy(w)

    return numpy.array([x,v])
