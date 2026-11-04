OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
x q[1];
cry(5.301437602932776) q[1],q[0];
cx q[0],q[1];
cry(3.730641276137879) q[3],q[5];
cx q[5],q[3];