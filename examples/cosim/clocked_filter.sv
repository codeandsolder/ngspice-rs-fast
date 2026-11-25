// SPDX-License-Identifier: BSD-3-Clause
module clocked_filter(input logic sample,
                      output logic clock = 0,
                      output logic filtered);
    timeunit 1ns;
    timeprecision 1ns;
    logic [2:0] history = 0;
    always #5 clock = ~clock;
    always @(posedge clock) history <= {history[1:0], sample};
    assign filtered = &history;
endmodule
