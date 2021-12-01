function mpc = WB3
%% 3 bus case in MATPOWER format
%  W. A. Bukhsh, Feb 2013 
%%   References:
%   [1] W. A. Bukhsh, Andreas Grothey, Ken McKinnon, Paul trodden, "Local Solutions of Optimal Power Flow Problem"
%       submitted to IEEE Transactions on Power Systems, 2013
%   [2] W. A. Bukhsh, Andreas Grothey, Ken McKinnon, Paul trodden, "Local Solutions of Optimal Power Flow Problem"
%       Technical Report ERGO, 2011
%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data
%       bus_i   type    Pd      Qd      Gs      Bs      area    Vm      Va      baseKV  zone    Vmax    Vmin
mpc.bus = [
        1       3       0          0       0    0       1  0.95         0     0     1       1.05    0.95;
        2       1       120        86.0    0    0       1  0.95        -55    0     1       1.05    0.95;
        3       1       68.0       50.0    0    0       1  0.95        -59    0     1       1.05    0.95;

];

%% generator data
%       bus     Pg      Qg      Qmax    Qmin    Vg        mBase status  Pmax    Pmin    Pc1     Pc2     Qc1min  Qc1max  Qc2min  Qc2max  ramp_agc        ramp_10 ramp_30 ramp_q  apf
mpc.gen = [
1       300     129     3000    -3000    1.1    100     1      5000    0       0       0  0 0  0 0  0  0  0  0 0;


];

%% branch data
%       fbus    tbus    r       x       b       rateA   rateB   rateC   ratio   angle   status  angmin  angmax
mpc.branch = [
        1       2       0.04    0.20    0.0       990000  0       0       0       0       1       -360    360;
        2       3       0.0139  0.0605  2.459     990000  0       0       0       0       1       -360    360;

];


%0.0139	0.0605	2.459
%%-----  OPF Data  -----%%
%% generator cost data
%       1       startup shutdown        n       x1      y1      ...     xn      yn
%       2       startup shutdown        n       c(n-1)  ...     c0
mpc.gencost = [
        2       0       0       3       0       2       0;

];
