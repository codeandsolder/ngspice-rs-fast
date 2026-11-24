//! Physical reciprocal coupled-line differential fixture, generated
//! by an independent Python matrix-exponential reference solver.

struct PhysicalFixture {
    name: &'static str,
    stages: usize,
    modes: usize,
    half_delays: &'static [f64],
    static_scattering: &'static [f64],
    earliest: f64,
    final_time: f64,
    steps: u32,
    checkpoints: &'static [(u32, &'static [f64], &'static [f64])],
}

static CASES: &[PhysicalFixture] = &[
    PhysicalFixture {
        name: "physical_scalar_RG",
        stages: 8,
        modes: 1,
        half_delays: &[f64::from_bits(0x3dd1_2e0b_e826_d696)],
        static_scattering: &[
            f64::from_bits(0xbf76_5f0a_fbdd_f979),
            f64::from_bits(0x3fef_c6fa_8c72_a480),
            f64::from_bits(0x3fef_c6fa_8c72_a47f),
            f64::from_bits(0xbf76_5f0a_fbdd_f980),
        ],
        earliest: f64::from_bits(0x3e11_2e0b_e826_d696),
        final_time: f64::from_bits(0x3e29_c511_dc3a_41e1),
        steps: 192,
        checkpoints: &[
            (
                1,
                &[f64::from_bits(0x0000_0000_0000_0000)],
                &[f64::from_bits(0x0000_0000_0000_0000)],
            ),
            (
                27,
                &[f64::from_bits(0xbf7b_8ab2_902c_67ec)],
                &[f64::from_bits(0x0000_0000_0000_0000)],
            ),
            (
                64,
                &[f64::from_bits(0xbf93_e636_7d5d_d78f)],
                &[f64::from_bits(0x0000_0000_0000_0000)],
            ),
            (
                96,
                &[f64::from_bits(0xbf9e_760f_0f09_c07e)],
                &[f64::from_bits(0x3fee_4468_3d7e_3986)],
            ),
            (
                128,
                &[f64::from_bits(0xbfa4_5e11_f085_92e1)],
                &[f64::from_bits(0x3fee_46e6_0faa_ed2b)],
            ),
            (
                160,
                &[f64::from_bits(0xbfa5_5276_b723_933c)],
                &[f64::from_bits(0x3fee_4871_1ab9_fa53)],
            ),
            (
                192,
                &[f64::from_bits(0xbfa5_5303_5d24_ed6f)],
                &[f64::from_bits(0x3fee_4916_2be4_b588)],
            ),
        ],
    },
    PhysicalFixture {
        name: "physical_noncommuting_three",
        stages: 8,
        modes: 3,
        half_delays: &[
            f64::from_bits(0x3dd5_9a5d_c66a_76a9),
            f64::from_bits(0x3ddd_b614_d410_309c),
            f64::from_bits(0x3de2_d795_2487_5dd4),
        ],
        static_scattering: &[
            f64::from_bits(0xbf97_2d73_aecf_0698),
            f64::from_bits(0x3f58_1e9e_ded9_13f1),
            f64::from_bits(0x3f49_b59e_894a_9407),
            f64::from_bits(0x3fef_41b5_6cf4_f314),
            f64::from_bits(0x3f57_8ac3_2ba5_9dd4),
            f64::from_bits(0x3f48_e3d1_1e1e_262d),
            f64::from_bits(0x3f58_1e9e_ded9_1482),
            f64::from_bits(0xbf8c_5883_f0aa_0935),
            f64::from_bits(0x3f4e_30ea_3a18_39d4),
            f64::from_bits(0x3f57_8ac3_2ba5_9dc5),
            f64::from_bits(0x3fef_8795_3b4f_b634),
            f64::from_bits(0x3f4d_aaf6_ccd7_efbc),
            f64::from_bits(0x3f49_b59e_894a_93c0),
            f64::from_bits(0x3f4e_30ea_3a18_3ade),
            f64::from_bits(0xbf79_ff32_3bb8_57dc),
            f64::from_bits(0x3f48_e3d1_1e1e_260e),
            f64::from_bits(0x3f4d_aaf6_ccd7_ef63),
            f64::from_bits(0x3fef_c2b2_0e16_1465),
            f64::from_bits(0x3fef_41b5_6cf4_f315),
            f64::from_bits(0x3f57_8ac3_2ba5_a158),
            f64::from_bits(0x3f48_e3d1_1e1e_257f),
            f64::from_bits(0xbf97_2d73_aecf_0700),
            f64::from_bits(0x3f58_1e9e_ded9_1485),
            f64::from_bits(0x3f49_b59e_894a_93e9),
            f64::from_bits(0x3f57_8ac3_2ba5_a167),
            f64::from_bits(0x3fef_8795_3b4f_b64d),
            f64::from_bits(0x3f4d_aaf6_ccd7_ee4e),
            f64::from_bits(0x3f58_1e9e_ded9_13f4),
            f64::from_bits(0xbf8c_5883_f0aa_09c0),
            f64::from_bits(0x3f4e_30ea_3a18_3b07),
            f64::from_bits(0x3f48_e3d1_1e1e_259c),
            f64::from_bits(0x3f4d_aaf6_ccd7_eeaa),
            f64::from_bits(0x3fef_c2b2_0e16_146c),
            f64::from_bits(0x3f49_b59e_894a_942e),
            f64::from_bits(0x3f4e_30ea_3a18_39ff),
            f64::from_bits(0xbf79_ff32_3bb8_5800),
        ],
        earliest: f64::from_bits(0x3e15_9a5d_c66a_76a9),
        final_time: f64::from_bits(0x3e30_33c6_54cf_d8ff),
        steps: 192,
        checkpoints: &[
            (
                1,
                &[
                    f64::from_bits(0x0000_0000_0000_0000),
                    f64::from_bits(0x0000_0000_0000_0000),
                    f64::from_bits(0x0000_0000_0000_0000),
                ],
                &[
                    f64::from_bits(0x0000_0000_0000_0000),
                    f64::from_bits(0x0000_0000_0000_0000),
                    f64::from_bits(0x0000_0000_0000_0000),
                ],
            ),
            (
                27,
                &[
                    f64::from_bits(0xbf9c_5c3f_9202_30a9),
                    f64::from_bits(0x3f58_0f05_1a85_33ac),
                    f64::from_bits(0x3f49_b55f_69f8_ed64),
                ],
                &[
                    f64::from_bits(0x0000_0000_0000_0000),
                    f64::from_bits(0x0000_0000_0000_0000),
                    f64::from_bits(0x0000_0000_0000_0000),
                ],
            ),
            (
                64,
                &[
                    f64::from_bits(0xbfb3_bef2_021d_3bdf),
                    f64::from_bits(0x3f70_bd98_e37f_e870),
                    f64::from_bits(0x3f60_82b0_257b_2b93),
                ],
                &[
                    f64::from_bits(0x3ca8_8c2d_186b_35d0),
                    f64::from_bits(0x0000_0000_0000_0000),
                    f64::from_bits(0x0000_0000_0000_0000),
                ],
            ),
            (
                96,
                &[
                    f64::from_bits(0xbfbd_52e0_5f93_e720),
                    f64::from_bits(0x3f79_4bb0_27a8_ae38),
                    f64::from_bits(0x3f67_5828_4291_d5ca),
                ],
                &[
                    f64::from_bits(0x3fea_9ccf_0c91_7c84),
                    f64::from_bits(0x3f83_b0c4_f5b8_2a82),
                    f64::from_bits(0x3f67_517b_6067_4be3),
                ],
            ),
            (
                128,
                &[
                    f64::from_bits(0xbfc3_0a8d_1dac_5d39),
                    f64::from_bits(0x3f7f_5d03_d514_47ff),
                    f64::from_bits(0x3f6f_145f_f768_2b0c),
                ],
                &[
                    f64::from_bits(0x3fea_c0cd_a8a8_bc12),
                    f64::from_bits(0x3f83_943b_ced0_83d9),
                    f64::from_bits(0x3f75_f829_51d3_7d44),
                ],
            ),
            (
                160,
                &[
                    f64::from_bits(0xbfc3_df49_552b_8ba1),
                    f64::from_bits(0x3f82_faf2_ffd4_0d27),
                    f64::from_bits(0x3f72_efb9_94c1_9de8),
                ],
                &[
                    f64::from_bits(0x3fea_d5da_8984_6449),
                    f64::from_bits(0x3f82_f838_519f_d036),
                    f64::from_bits(0x3f75_5aa5_6afc_9e2c),
                ],
            ),
            (
                192,
                &[
                    f64::from_bits(0xbfc3_e6b6_e1e2_399c),
                    f64::from_bits(0x3f82_f265_d43a_d1e2),
                    f64::from_bits(0x3f75_6b1d_30dd_4bac),
                ],
                &[
                    f64::from_bits(0x3fea_de4d_e10d_d974),
                    f64::from_bits(0x3f82_9510_7d05_30e0),
                    f64::from_bits(0x3f74_edf2_df12_ba7d),
                ],
            ),
        ],
    },
];

#[test]
fn physical_rlgc_scattering_matches_independent_python_accepted_history() -> Result<(), &'static str>
{
    use super::WaveNetwork;
    for fixture in CASES {
        let mut network = WaveNetwork::new(
            fixture.half_delays.to_vec(),
            fixture.stages,
            fixture.static_scattering.to_vec(),
            vec![0.; fixture.modes],
            vec![f64::INFINITY; fixture.modes],
            true,
        )?;
        let mut check = 0;
        for step in 1..=fixture.steps {
            let time = fixture.final_time * (f64::from(step) / f64::from(fixture.steps));
            let mut incident = vec![0.; fixture.modes];
            incident[0] = (time / (0.2 * fixture.earliest)).min(1.);
            let trial = network.trial(time, &incident, &vec![0.; fixture.modes])?;
            if check < fixture.checkpoints.len() {
                let (wanted_step, expected_near, expected_far) = fixture.checkpoints[check];
                if step == wanted_step {
                    for (&actual, &expected) in trial.reflected_near().iter().zip(expected_near) {
                        assert!(
                            (actual - expected).abs() < 5e-10,
                            "{} near mismatch at step {}: {} vs {}",
                            fixture.name,
                            step,
                            actual,
                            expected
                        );
                    }
                    for (&actual, &expected) in trial.transmitted_far().iter().zip(expected_far) {
                        assert!(
                            (actual - expected).abs() < 5e-10,
                            "{} far mismatch at step {}: {} vs {}",
                            fixture.name,
                            step,
                            actual,
                            expected
                        );
                    }
                    check += 1;
                }
            }
            network.accept(trial)?;
        }
        assert_eq!(check, fixture.checkpoints.len());
        let (input, output) = network.energies();
        assert!(
            output <= input,
            "{} violated power-wave energy passivity",
            fixture.name
        );
    }
    Ok(())
}
