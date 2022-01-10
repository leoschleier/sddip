function mpc = LMBM3
%% case3_bernie    Power flow data for 3 bus, 3 generator case.

% This test case is taken from the following paper

%  Lesieutre, B.C., Molzahn, D.K., Borden, A.R., DeMarco, C.L., "Examining the
%  limits of the application of semidefinite programming to power flow problems"
%  49th Annual Allerton Conference on Communication, Control, and Computing, 2011

%   Please see CASEFORMAT for details on the case file format.
%% W. A. Buksh, April 2013
% w.a.bukhsh@sms.ed.ac.uk
%% MATPOWER Case Format : Version 2
mpc.version = '2';

%%-----  Power Flow Data  -----%%
%% system MVA base
mpc.baseMVA = 100;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	3	110	40	0	0	1	1.069	0       345	1	1.10	0.90;
	2	2	110 40	0	0	1	1.028	9.916	345	1	1.10	0.90;
	3	2	95	50	0	0	1	1.001	-13.561	345	1	1.10	0.90;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin	Pc1	Pc2	Qc1min	Qc1max	Qc2min	Qc2max	ramp_agc	ramp_10	ramp_30	ramp_q	apf
mpc.gen = [
	1	131.09	17.02	10000	-1000	1.069	100	1	10000	0	0	0	0	0	0	0	0	0	0	0	0;
    2	185.93	-3.50	1000	-1000	1.028	100	1	10000	0	0	0	0	0	0	0	0	0	0	0	0;
	3	0       0.06	1000	-1000	1.001	100	1	0   	0	0	0	0	0	0	0	0	0	0	0	0;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	3	0.065	0.620   0.450	9999	9999	9999	0	0	1	-360	360;
    3	2   0.025	0.750   0.700	186  	9999	9999	0	0	1	-360	360;
	1	2	0.042	0.900	0.300	9999	9999	9999	0	0	1	-360	360;
];

%% generator cost data
%	1	startup	shutdown	n	x1	y1	...	xn	yn
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	0	0	3	0.110	5	0;
	2	0	0	3	0.085	1.2	0;
    2	0	0	3	0   	1	0;
];
