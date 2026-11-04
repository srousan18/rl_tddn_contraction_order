OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
x q[2];
cry(3*pi/2) q[2],q[4];
cx q[4],q[2];
cry(6.086835766330224) q[5],q[1];
cx q[1],q[5];