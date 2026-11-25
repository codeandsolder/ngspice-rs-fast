-- SPDX-License-Identifier: BSD-3-Clause
library ieee;
use ieee.std_logic_1164.all;

entity clocked_filter is
    port (
        sample : in std_logic;
        filtered : out std_logic;
        clock : out std_logic
    );
end entity;

architecture rtl of clocked_filter is
    signal internal_clock : std_logic := '0';
    signal history : std_logic_vector(2 downto 0) := "000";
begin
    internal_clock <= not internal_clock after 5 ns;
    clock <= internal_clock;
    process(internal_clock)
    begin
        if rising_edge(internal_clock) then
            history <= history(1 downto 0) & sample;
        end if;
    end process;
    filtered <= history(2) and history(1) and history(0);
end architecture;
