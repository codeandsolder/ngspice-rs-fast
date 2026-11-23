# Streaming rawfile output: single-pass atomic writer (#142)

Implementation: src/analysis/rawfile/mod.rs (RawFileWriter), transaction.rs (RawFileTransaction) and binary.rs.

## Output contract

The CLI exclusively creates one temporary file beside the destination, writes retained and streamed plots into that same file, then atomically replaces the output only after successful completion. Direct RawFileWriter::create is a non-atomic lower-level writer; use RawFileTransaction for atomic replacement. There is no separate rawfile point spool or second payload copy.

Retained/complete plots keep their historical compact No. Points: N header and exact byte encoding. For an initially unknown count:

1. Validate the column schema; clear retained rows. Render the common header prefix and reserve **exactly 20 bytes** after No. Points: followed by a newline.
2. Record the seekable output offset. Write the common variable declarations and Values: or Binary: marker.
3. Encode each validated borrowed row directly to the same temporary file, using one bounded 32 KiB binary staging block and one buffered output file. Count increments are checked.
4. Flush the final binary block, obtain the stream end position, seek to the reserved field, overwrite **all 20 bytes** with right-aligned decimal count, then restore the end position for the next plot.

The unsigned 64-bit maximum has 20 decimal digits. Count width is bounded and formatted with an explicit checked reservation. Left whitespace is accepted by the C rawfile numeric scanner and by this port's header parser (which trims it before parse_count). No byte needs to move and the length of the header never changes.

The header's padded count is the **only intentional byte difference** between equivalent retained and streamed output. The original captured C golden files, retained plot encoding, payload values and order, real/complex representation, signed zeros, binary endianness, and variable declarations are unchanged. Removing count padding after writing would require reading and copying the entire payload and is intentionally avoided.

## Failure lifecycle

RawFileWriter holds at most one active streamed plot. State, schema, count, write, flush or seek error permanently poisons it. Finishing an incomplete or poisoned writer is rejected. RawFileTransaction also latches errors and cannot publish incomplete or failed output. A successfully finished writer flushes before its exclusively owned temporary is atomically renamed to the destination. An unpublished temporary is removed on drop after inode verification; collisions never truncate or delete someone else's file.

Successive retained and streamed plots reuse the same file and encoding state. A 0-point plot has a zero count and no rows. A rootless one-row, zero-column PZ plot preserves C's ASCII index/TAB without a following newline, or zero binary payload bytes. These unusual shapes are explicitly tested for both formats and for multi-plot concatenation. The pinned ngspice-47+ reader accepts ordinary padded-count ASCII/binary, including 0-point plots, but **cannot materialize zero-column rootless PZ as a loaded plot**: it rejects its own historical ASCII 0+TAB row as a strange rawfile line and finds no data in the empty binary payload. The live-C test exercises and records this C reader limitation; the Rust reader preserves the legitimate one-row PZ shape.

The low-level writer now directly exposes begin_plot / push_point / finish_plot; the old transferable spool-owning StreamedPlot object has been removed deliberately. RawFileTransaction retains its existing direct stream API.

## Bounded payload I/O

For **P** bytes of plot payload (excluding headers):

| Pipeline | Payload bytes written | Payload bytes read | Other work |
| --- | ---: | ---: | --- |
| Prior separate spool + replay | 2P | P | Two files; complete payload copy |
| Direct atomic streaming | P | 0 | One file; two count patch seeks and 20-byte overwrite |

These are logical I/O counts. Actual physical storage I/O is affected by page cache and concurrent host activity, so elapsed time is separately measured rather than inferred.

Acer 2026-11-23, 131,072 rows × 16 real columns = 16 MiB of payload, three alternating repetitions with a busy concurrent build host:

| Pipeline | Trial 1 (ms) | Trial 2 (ms) | Trial 3 (ms) |
| --- | ---: | ---: | ---: |
| Direct | 27.148 | 11.678 | 11.777 |
| Historical-style spool write + replay | 1968.670 | 85.157 | 68.249 |

The first replay trial is a cold/outlier under host I/O pressure; warm direct trials are approximately 11.7 ms and replay 68–85 ms. This benchmark isolates writing and copying the payload, not the SPICE solver. The historical-style replay writes one 128-byte row at a time through BufWriter, as the old per-row flushed binary encoder did, while direct output accumulates 32 KiB blocks; the speedup includes both the eliminated copy and changed block coalescing. The comparison does not claim identical syscall traces or physical SSD writes.

Memory is independent of row count: binary staging at most 32 KiB, default BufWriter 8 KiB and the caller's borrowed row. No complete plot is held in memory. A repeat 16 MiB combined direct/replay run measured 2,776 KiB peak process RSS using /usr/bin/time -v; this is a single-process maximum, not an isolated per-pipeline heap allocation count.

## Validation

The core rawfile unit tests cover backpatching, atomic collision/replacement, poisoning and cleanup, zero-width and malformed rows, plus seek/nonseekable and flush failure injection. The streamed_output integration test verifies mixed multi-plot byte identity after removing only the reserved count padding, including binary little/big endian. The opt-in rawfile_streaming_bench integration test runs a 16 MiB storage workload with alternating direct/replay trials, and exercises C's load command with both ASCII and native-endian binary output, 0-point files, and the pinned C reader limitation for rootless-PZ files.

To reproduce the opt-in C check, set NGSPICE_BIN to the absolute C ngspice binary and run the ignored c_reads_backpatched_counts_in_both_formats test in rawfile_streaming_bench. The benchmark is the ignored direct_io_work_replaces_two_spool_writes_and_one_payload_read test. Both use self-cleaning scratch directories.

No golden capture or comparison tolerance changes are required.
