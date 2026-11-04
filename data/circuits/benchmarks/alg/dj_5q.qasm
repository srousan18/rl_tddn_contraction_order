OPENQASM 2.0;
include "qelib1.inc";
gate gate_Oracle q0,q1,q2,q3,q4 { x q0; x q1; cx q0,q4; cx q1,q4; cx q2,q4; cx q3,q4; x q0; x q1; }
qreg q[5];
h q[0];
h q[1];
h q[2];
h q[3];
x q[4];
h q[4];
gate_Oracle q[0],q[1],q[2],q[3],q[4];
h q[0];
h q[1];
h q[2];
h q[3];