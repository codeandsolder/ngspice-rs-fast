-- SPDX-License-Identifier: BSD-3-Clause
library ieee;
use ieee.std_logic_1164.all;

entity feedback_inverter is
    port (sense : in std_logic; control : out std_logic := '1');
end entity;

architecture rtl of feedback_inverter is
begin
    control <= not sense after 500 ns;
end architecture;
