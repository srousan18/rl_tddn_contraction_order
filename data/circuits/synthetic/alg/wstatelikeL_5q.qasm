OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
x q[2];
cry(3*pi/2) q[2],q[4];
cx q[4],q[2];
cry(7*pi/16) q[3],q[1];
cx q[1],q[3];