OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
x q[3];
cry(pi/4) q[1],q[4];
cx q[4],q[1];
cry(5*pi/16) q[4],q[0];
cx q[0],q[4];