// SPDX-License-Identifier: BSD-3-Clause
module feedback_inverter(input logic sense, output logic control = 1);
    timeunit 1ns;
    timeprecision 1ns;
    always @(sense) control <= #500 ~sense;
endmodule
