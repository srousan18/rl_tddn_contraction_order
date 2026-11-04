OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
x q[1];
cry(5.301437602932776) q[4],q[0];
cx q[0],q[4];
cry(pi/8) q[0],q[2];
cx q[2],q[0];