# QC-QLDPC-Decoders
This repository contains the MatLab programs to simulate the results published in the paper "Entanglement-Assisted Quasi-cyclic Quantum Low-density Parity-check Codes over Qubits" authored by Pavan Kumar, Abhi Kumar Sharma, Karthik Bharadwaj, and Shayan Srinivasa Garani, published in Quantum.

README
======

PROGRAM FILES
-------------
These are the main scripts you can run directly. Each one produces a specific
figure from the paper.

bin_vs_quat_p11.m

  Run this to generate Figure 8a from the paper. It compares four decoding methods
  — BLSP, QMS, QNMS, and QBLNMS — for the code [[121,20,10;1]]_2, both when burst
  errors are present and when they are not.

bin_vs_quat_p13.m

  Run this to generate Figure 8b from the paper. It does the same comparison as
  above but for a different code: [[169,24,12;1]]_2.

comp_hagiwara.m

  Run this to generate Figure 10 from the paper. It compares the error correction
  performance of two codes: the [[42,4]]_2 code (proposed by Hagiwara et al.) and
  our [[42,10;6]]_2 code, both using the QBLNMS decoder.

comp_hsieh.m

  Run this to generate Figure 11 from the paper. It compares the performance of
  the [[128,58;18]]_2 code (proposed by Min-Hsieh et al.) against our
  [[121,70;51]]_2 code, both using the QBLNMS decoder.
  
Gg4_vs_Gg6.m

  Run this to generate Figure 12 from the paper. It compares two codes:
  [[390,132;128]]_2 (which has girth greater than 6) and [[529,176;353]]_2 (which
  has girth greater than 4), both decoded using the QBLNMS decoder.

cw_comp.m

  Run this to generate Figure 13 from the paper. It shows how the performance of
  the [[289,k;1]]_2 code changes as the column weight varies from 3 to 8, using
  the QBLNMS decoder.



SUPPORTING FUNCTION FILES
--------------------------
These files are helper functions used internally by the program files above.
You do not run these directly.

Encoder_shift.m

  Generates the parity-check matrices for the codes using a shift-based
  construction method.

Soft_Dec_bsp_lay.m

  Implements the Binary Layered Sum-Product (BLSP) decoding algorithm. It operates on
  individual Tanner graphs, one for each component code.

Soft_Dec_combine.m

  Implements the Quaternary Min-Sum (QMS) decoding algorithm. It works on a
  combined (joint) Tanner graph that handles both component codes together.

Soft_Dec_combine_NMS.m

  Implements the Quaternary Normalized Min-Sum (QNMS) decoding algorithm — an
  improved version of QMS that scales the messages to get better performance.

Soft_Dec_combine_BLNMS_dyn.m

  Implements the Quaternary Block-Layered Normalized Min-Sum (QBLNMS) decoding algorithm
  — the most refined version, which processes the block-layers with dynamic alpha/beta 
  adaptation for each block layer in each iteration for faster convergence and better results.



OTHER FILES
-----------
4_cycles_in_CSS.pdf 
- This pdf includes the proof for the inevitability of 4-cycles in Tanner graphs of CSS codes
