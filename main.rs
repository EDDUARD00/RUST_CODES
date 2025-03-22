mod LPcalc; // Computes the LP modes of cavity
mod it; // Generates the current and time vectors
mod injprof; // Current spreading function
mod dij; // Kronecker deltas
mod abcdef; // Computes the parameters of the algorithm
mod params; // Contains the VCSEL multimode parameters

use injprof::{injprof, trapz};
use LPcalc::lp_calc;
use it::generate_current_vectors;
use abcdef::compute_abcdef;
use params::MMParams;
use nalgebra::{DMatrix, DVector};
use ndarray::Array1;
use std::time::Instant;

// Import CSV writing functionality
use std::fs::File;
use std::io::Write;
use std::io::Result;

/// Saves a matrix to a CSV file
fn save_to_csv(filename: &str, data: &DMatrix<f64>) -> Result<()> {
    let mut file = File::create(filename)?;
    for row in data.row_iter() {
        let row_str: Vec<String> = row.iter().map(|&x| x.to_string()).collect();
        writeln!(file, "{}", row_str.join(","))?;
    }
    Ok(())
}

fn vistas_basic() -> std::io::Result<()> {
    let start_time = Instant::now();

    // Initialize VCSEL parameters
    let params = MMParams::new();
    let hvl = params.h * params.c0 / 1e-9;
    let vol = std::f64::consts::PI * params.R.powi(2) * (params.nw as f64 * params.dqw);
    let vg = 100.0 * params.c0 / params.ng;

    // Compute mirror loss and photon lifetime
    let alpha_m = (1.0f64 / params.L) * (1.0f64 / (params.Rtc[0] * params.Rbc[0]).sqrt()).ln();
    let tau_s = 1.0 / vg / (alpha_m + params.alpha_ic[0]);
    let f = (1.0 - params.Rtc[0]) / ((1.0 - params.Rtc[0]) + (params.Rtc[0] / params.Rbc[0]).sqrt() * (1.0 - params.Rbc[0]));
    let eta_opt = f * alpha_m / (params.alpha_ic[0] + alpha_m);

    // Generate radial vector
    let ni = 10;
    let r_values: DVector<f64> = DVector::from_vec((0..=ni).map(|i| i as f64 * params.R / ni as f64).collect());
    let cr = r_values.len() - 1;

    // Generate time and current vectors
    let (modformat, ct, ctinit, t, tb, i0) = generate_current_vectors(params.dt);

    // Convert DVector to Array1 for injprof
    let r_values_array = Array1::from_vec(r_values.as_slice().to_vec());

    // Generate current spreading function
    let (fcr, fcphi) = injprof(&r_values_array, params.R, params.Rox, params.rs, params.dphi);

    // Generate LP modes
    let (nm, lplm, lvec, ur) = lp_calc(params.lfp0 * 1e-7, params.nc, params.dn, params.Rm, params.R, cr);

    // Normalize the modal intensities
    let mut ur_sq = ur.clone();
    for k in 0..nm {
        let y = Array1::from_vec(ur.row(k).transpose().as_slice().to_vec());
        let normalization = trapz(&r_values_array, &y);
        ur_sq.row_mut(k).apply(|x| *x = x.powi(2) * params.R.powi(2) / (2.0 * normalization));
    }

    // Initialize matrices for carrier density and photon density
    let mut n = DMatrix::zeros(ni, ct + 1);
    let mut s = DMatrix::zeros(nm, ct + 1);
    let mut g = DMatrix::from_element(nm, ni, 0.0);

    // Convert Array1 to DVector for compute_abcdef
    let fcr_dvector = DVector::from_vec(fcr.to_vec());
    let fcphi_dvector = DVector::from_vec(fcphi.to_vec());

    // Compute algorithm parameters
    let (a, b, cc, cs, drad, dang, e, fcc, fcs, fsc, fss) = compute_abcdef(
        &lvec,
        &r_values,
        params.R,
        params.Rm,
        nm,
        ni,
        1,
        &fcr_dvector,
        &fcphi_dvector,
        &ur_sq,
    );

    // Update carrier and photon densities
    let b = b.component_mul(&(params.Gamma.component_mul(&params.beta) / params.tau_n));
    let c = cc * params.eta_i / params.q / vol;
    let drad = params.DN * drad + DVector::from_element(drad.len(), 1.0 / params.tau_n);
    let e = vg * e;

    for i in 0..ct {
        let n_tmp = n.column(i);
        let s_tmp = s.column(i);
        let rsp = &b * &n_tmp;

        let g0 = params.gln * ((n_tmp.sum() + 1.0) / params.Ntr).ln() / (n_tmp.sum() - params.Ntr);

        for m in 0..nm {
            let etmp = g0 * &e.row(m);
            g.row_mut(m).zip_apply(&n_tmp.transpose(), |x, y| *x = etmp * y);
        }

        let s2 = s_tmp.map(|x| x / tau_s);
        g.zip_apply(&s2, |x, y| *x *= y);
        
        //let drad_sum = drad.sum();
        //let g_sum= g.sum();

        n.column_mut(i + 1).zip_apply(&n_tmp, |x, y| *x = y + params.dt * (c * i0[i] - y * drad - g.sum()));
        s.column_mut(i + 1).zip_apply(&s_tmp, |x, y| *x = y + params.dt * (-y / tau_s + params.Gamma[0] * g.column(0)[0] + rsp[0]));
    }

    // Compute output power
    let kcs = vol * hvl / params.lfp0 * eta_opt / tau_s / params.Gamma[0];
    let pcs = &s * kcs * 1e3;

    // Print elapsed time
    let elapsed = start_time.elapsed();
    println!("Main calculation = {:?}s", elapsed.as_secs_f64());

    // Save simulation results to CSV files
    save_to_csv("time.csv", &DMatrix::from_fn(1, t.len(), |_, j| t[j]))?;
    save_to_csv("optical_power.csv", &pcs)?;
    save_to_csv("carrier_density.csv", &n)?;
    save_to_csv("modal_intensities.csv", &ur_sq)?;

    Ok(())
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    vistas_basic()?;
    Ok(())
}