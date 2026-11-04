OPENQASM 2.0;
include "qelib1.inc";
gate gate_Oracle q0,q1,q2,q3,q4,q5 { x q0; x q1; x q4; cx q0,q5; cx q1,q5; cx q2,q5; cx q3,q5; cx q4,q5; x q0; x q1; x q4; }
qreg q[6];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
x q[5];
h q[5];
gate_Oracle q[0],q[1],q[2],q[3],q[4],q[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];