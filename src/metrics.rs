use core_foundation::dictionary::CFDictionaryRef;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::sources::{
  cfio_bytes, cfio_get_residencies, cfio_watts, libc_ram, libc_swap, IOHIDSensors, IOReport,
  SocInfo, SMC,
};

type WithError<T> = Result<T, Box<dyn std::error::Error>>;

// const CPU_FREQ_DICE_SUBG: &str = "CPU Complex Performance States";
const CPU_FREQ_CORE_SUBG: &str = "CPU Core Performance States";
const GPU_FREQ_DICE_SUBG: &str = "GPU Performance States";

// MARK: Structs

#[derive(Debug, Default)]
pub struct TempMetrics {
  pub cpu_temp_avg: f32, // Celsius
  pub gpu_temp_avg: f32, // Celsius
}

#[derive(Debug, Default)]
pub struct MemMetrics {
  pub ram_total: u64,  // bytes
  pub ram_usage: u64,  // bytes
  pub swap_total: u64, // bytes
  pub swap_usage: u64, // bytes
}

#[derive(Debug, Default, Clone)]
pub struct MetricHistogram {
  pub labels: Vec<String>,
  pub values: Vec<u64>,
}

impl MetricHistogram {
  pub fn weighted_average_gbps(&self, label: &str) -> f64 {
    log_to_file(&format!("\nCalculating weighted average bandwidth for {}:", label));
     let mut sum_products = 0.0;
    let sum_weights: u64 = self.values.iter().sum();

    log_to_file(&format!("Total weights (samples): {}", sum_weights));

    for (label, &weight) in self.labels.iter().zip(self.values.iter()) {
      if let Ok(gbps) = label.trim().replace("GB/s", "").trim().parse::<f64>() {
        let contribution = gbps * (weight as f64);
        log_to_file(&format!("Band {} GB/s: {} samples, contribution: {}", 
          gbps, weight, contribution));
        sum_products += contribution;
      }
    }

    let avg = if sum_weights > 0 {
      sum_products / (sum_weights as f64)
    } else {
      0.0
    };

    log_to_file(&format!("Weighted average: {} GB/s", avg));
    avg
  }
}

#[derive(Debug, Default)]
pub struct Metrics {
  pub temp: TempMetrics,
  pub memory: MemMetrics,
  pub ecpu_usage: (u32, f32),       // freq, percent_from_max
  pub pcpu_usage: (u32, f32),       // freq, percent_from_max
  pub gpu_usage: (u32, f32),        // freq, percent_from_max
  pub cpu_power: f32,               // Watts
  pub gpu_power: f32,               // Watts
  pub ane_power: f32,               // Watts
  pub all_power: f32,               // Watts
  pub sys_power: f32,               // Watts
  pub ane_rd_bw: u64,               // ANE read bandwidth in B/s
  pub ane_wr_bw: u64,               // ANE write bandwidth in B/s
  pub sys_rd_bw: u64,               // system read bandwidth in B/s (excluding ANE)
  pub sys_wr_bw: u64,               // system write bandwidth in B/s (excluding ANE)
//  pub ane_bw_rd_hist: MetricHistogram, // GB/s distribution of ANE read+write BW
  pub ane_bw_rd_hist: MetricHistogram, // GB/s distribution of ANE read BW
  pub ane_bw_wr_hist: MetricHistogram, // GB/s distribution of ANE read BW


  pub sys_rd_bw_hist: MetricHistogram, // GB/s distribution of system read BW
  pub sys_rw_bw_hist: MetricHistogram, // GB/s distribution of system read+write BW
  pub floor_rd_bw_hist: MetricHistogram, // GB/s distribution of system 
  pub floor_rw_bw_hist: MetricHistogram, // GB/s distribution of system 


}

// MARK: Helpers

pub fn zero_div<T: core::ops::Div<Output = T> + Default + PartialEq>(a: T, b: T) -> T {
  let zero: T = Default::default();
  return if b == zero { zero } else { a / b };
}

fn calc_freq(item: CFDictionaryRef, freqs: &Vec<u32>) -> (u32, f32) {
  let items = cfio_get_residencies(item); // (ns, freq)
  let (len1, len2) = (items.len(), freqs.len());
  assert!(len1 > len2, "cacl_freq invalid data: {} vs {}", len1, len2); // todo?

  // IDLE / DOWN for CPU; OFF for GPU; DOWN only on M2?/M3 Max Chips
  let offset = items.iter().position(|x| x.0 != "IDLE" && x.0 != "DOWN" && x.0 != "OFF").unwrap();

  let usage = items.iter().map(|x| x.1 as f64).skip(offset).sum::<f64>();
  let total = items.iter().map(|x| x.1 as f64).sum::<f64>();
  let count = freqs.len();

  let mut avg_freq = 0f64;
  for i in 0..count {
    let percent = zero_div(items[i + offset].1 as _, usage);
    avg_freq += percent * freqs[i] as f64;
  }

  let usage_ratio = zero_div(usage, total);
  let min_freq = freqs.first().unwrap().clone() as f64;
  let max_freq = freqs.last().unwrap().clone() as f64;
  let from_max = (avg_freq.max(min_freq) * usage_ratio) / max_freq;

  (avg_freq as u32, from_max as f32)
}

fn calc_freq_final(items: &Vec<(u32, f32)>, freqs: &Vec<u32>) -> (u32, f32) {
  let avg_freq = zero_div(items.iter().map(|x| x.0 as f32).sum(), items.len() as f32);
  let avg_perc = zero_div(items.iter().map(|x| x.1 as f32).sum(), items.len() as f32);
  let min_freq = freqs.first().unwrap().clone() as f32;

  (avg_freq.max(min_freq) as u32, avg_perc)
}

fn sum_hist(a: &MetricHistogram, b: &MetricHistogram) -> MetricHistogram {
  log_to_file("\nSumming histograms:");
  log_to_file(&format!("Histogram A labels: {:?}", a.labels));
  log_to_file(&format!("Histogram A values: {:?}", a.values));
  log_to_file(&format!("Histogram B labels: {:?}", b.labels));
  log_to_file(&format!("Histogram B values: {:?}", b.values));

  assert!(a.labels == b.labels, "Histogram labels mismatch");
  let mut rs = MetricHistogram {
    labels: a.labels.clone(),
    values: vec![0; a.values.len()],
  };

  for i in 0..a.values.len() {
    rs.values[i] = a.values[i] + b.values[i];
  }

  log_to_file(&format!("Result histogram labels: {:?}", rs.labels));
  log_to_file(&format!("Result histogram values: {:?}", rs.values));

  rs
}

fn zero_div_hist(a: &MetricHistogram, b: u64) -> MetricHistogram {
  let values = a.values.iter().map(|x| zero_div(*x, b)).collect();
  MetricHistogram { labels: a.labels.clone(), values }
}

fn init_smc() -> WithError<(SMC, Vec<String>, Vec<String>)> {
  let mut smc = SMC::new()?;

  let mut cpu_sensors = Vec::new();
  let mut gpu_sensors = Vec::new();

  let names = smc.read_all_keys().unwrap_or(vec![]);
  for name in &names {
    let key = match smc.read_key_info(&name) {
      Ok(key) => key,
      Err(_) => continue,
    };

    if key.data_size != 4 || key.data_type != 1718383648 {
      continue;
    }

    let _ = match smc.read_val(&name) {
      Ok(val) => val,
      Err(_) => continue,
    };

    // Unfortunately, it is not known which keys are responsible for what.
    // Basically in the code that can be found publicly "Tp" is used for CPU and "Tg" for GPU.

    match name {
      name if name.starts_with("Tp") => cpu_sensors.push(name.clone()),
      name if name.starts_with("Tg") => gpu_sensors.push(name.clone()),
      _ => (),
    }
  }

  // println!("{} {}", cpu_sensors.len(), gpu_sensors.len());
  Ok((smc, cpu_sensors, gpu_sensors))
}

// MARK: Sampler

pub struct Sampler {
  soc: SocInfo,
  ior: IOReport,
  hid: IOHIDSensors,
  smc: SMC,
  smc_cpu_keys: Vec<String>,
  smc_gpu_keys: Vec<String>,
}

impl Sampler {
  pub fn new() -> WithError<Self> {
    let channels = vec![
      ("Energy Model", None), // cpu/gpu/ane power
      // ("CPU Stats", Some(CPU_FREQ_DICE_SUBG)), // cpu freq by cluster
      ("CPU Stats", Some(CPU_FREQ_CORE_SUBG)), // cpu freq per core
      ("GPU Stats", Some(GPU_FREQ_DICE_SUBG)), // gpu freq
      ("AMC Stats", Some("Perf Counters")),    // mem bw
      ("PMP", Some("AF BW")),                  // "Apple Fabric" (?) mem bw histogram
      ("PMP", Some("DCS BW")),                  // "Apple Fabric" (?) mem bw histogram
      ("PMP", Some("AFR BW")),                  // "Apple Fabric" (?) mem bw histogram
      ("PMP", Some("AFI BW")),                  // "Apple Fabric" (?) mem bw histogram
    ];

    let soc = SocInfo::new()?;
    let ior = IOReport::new(channels)?;
    let hid = IOHIDSensors::new()?;
    let (smc, smc_cpu_keys, smc_gpu_keys) = init_smc()?;

    Ok(Sampler { soc, ior, hid, smc, smc_cpu_keys, smc_gpu_keys })
  }

  fn get_temp_smc(&mut self) -> WithError<TempMetrics> {
    let mut cpu_metrics = Vec::new();
    for sensor in &self.smc_cpu_keys {
      let val = self.smc.read_val(sensor)?;
      let val = f32::from_le_bytes(val.data[0..4].try_into().unwrap());
      cpu_metrics.push(val);
    }

    let mut gpu_metrics = Vec::new();
    for sensor in &self.smc_gpu_keys {
      let val = self.smc.read_val(sensor)?;
      let val = f32::from_le_bytes(val.data[0..4].try_into().unwrap());
      gpu_metrics.push(val);
    }

    let cpu_temp_avg = zero_div(cpu_metrics.iter().sum::<f32>(), cpu_metrics.len() as f32);
    let gpu_temp_avg = zero_div(gpu_metrics.iter().sum::<f32>(), gpu_metrics.len() as f32);

    Ok(TempMetrics { cpu_temp_avg, gpu_temp_avg })
  }

  fn get_temp_hid(&mut self) -> WithError<TempMetrics> {
    let metrics = self.hid.get_metrics();

    let mut cpu_values = Vec::new();
    let mut gpu_values = Vec::new();

    for (name, value) in &metrics {
      if name.starts_with("pACC MTR Temp Sensor") || name.starts_with("eACC MTR Temp Sensor") {
        // println!("{}: {}", name, value);
        cpu_values.push(*value);
        continue;
      }

      if name.starts_with("GPU MTR Temp Sensor") {
        // println!("{}: {}", name, value);
        gpu_values.push(*value);
        continue;
      }
    }

    let cpu_temp_avg = zero_div(cpu_values.iter().sum(), cpu_values.len() as f32);
    let gpu_temp_avg = zero_div(gpu_values.iter().sum(), gpu_values.len() as f32);

    Ok(TempMetrics { cpu_temp_avg, gpu_temp_avg })
  }

  fn get_temp(&mut self) -> WithError<TempMetrics> {
    // HID for M1, SMC for M2/M3
    // UPD: Looks like HID/SMC related to OS version, not to the chip (SMC available from macOS 14)
    match self.smc_cpu_keys.len() > 0 {
      true => self.get_temp_smc(),
      false => self.get_temp_hid(),
    }
  }

  fn get_mem(&mut self) -> WithError<MemMetrics> {
    let (ram_usage, ram_total) = libc_ram()?;
    let (swap_usage, swap_total) = libc_swap()?;
    Ok(MemMetrics { ram_total, ram_usage, swap_total, swap_usage })
  }

  fn get_sys_power(&mut self) -> WithError<f32> {
    let val = self.smc.read_val("PSTR")?;
    let val = f32::from_le_bytes(val.data.clone().try_into().unwrap());
    Ok(val)
  }

  pub fn get_metrics(&mut self, duration: u64) -> WithError<Metrics> {
    let measures: usize = 4;
    let mut results: Vec<Metrics> = Vec::with_capacity(measures);

    // do several samples to smooth metrics
    // see: https://github.com/vladkens/macmon/issues/10
    for (sample, sample_dt) in self.ior.get_samples(duration, measures) {
      let mut ecpu_usages = Vec::new();
      let mut pcpu_usages = Vec::new();
      let mut rs = Metrics::default();

      for x in sample {
        if x.group == "CPU Stats" && x.subgroup == CPU_FREQ_CORE_SUBG {
          if x.channel.contains("ECPU") {
            ecpu_usages.push(calc_freq(x.item, &self.soc.ecpu_freqs));
            continue;
          }

          if x.channel.contains("PCPU") {
            pcpu_usages.push(calc_freq(x.item, &self.soc.pcpu_freqs));
            continue;
          }
        }

        if x.group == "GPU Stats" && x.subgroup == GPU_FREQ_DICE_SUBG {
          match x.channel.as_str() {
            "GPUPH" => rs.gpu_usage = calc_freq(x.item, &self.soc.gpu_freqs[1..].to_vec()),
            _ => {}
          }
        }

        if x.group == "Energy Model" {
          match x.channel.as_str() {
            "CPU Energy" => rs.cpu_power += cfio_watts(x.item, &x.unit, sample_dt)?,
            "GPU Energy" => rs.gpu_power += cfio_watts(x.item, &x.unit, sample_dt)?,
            c if c.starts_with("ANE") => rs.ane_power += cfio_watts(x.item, &x.unit, sample_dt)?,
            _ => {}
          }
        }

        if x.group == "AMC Stats" && x.subgroup == "Perf Counters" {
          log_to_file(&format!("Found AMC Stats channel: {}", x.channel));
          
          match x.channel.as_str() {
            // There are two values for ANE: "ANE0 DCS RD" and "ANE0 RD".
            // Per asitop, DCS = DRAM Command Scheduler. When asitop
            // supported bandwidth, it used the DCS values (for CPU).
            // The DCS value also seems closer to the number I expect for a test
            // ANE model based on the size of the weights and the tokens/second.
            // The non-DCS value is slightly higher.

            s if s.starts_with("ANE") && s.ends_with("DCS RD") => {
              let bytes = cfio_bytes(x.item);
              log_to_file(&format!("ANE DCS RD bytes: {}", bytes));
              rs.ane_rd_bw += bytes;
            }
            s if s.starts_with("ANE") && s.ends_with("DCS WR") => {
              let bytes = cfio_bytes(x.item);
              log_to_file(&format!("ANE DCS WR bytes: {}", bytes));
              rs.ane_wr_bw += bytes;
            }
            "DCS RD" => {
              let bytes = cfio_bytes(x.item);
              log_to_file(&format!("System DCS RD bytes: {}", bytes));
              rs.sys_rd_bw += bytes;
            }
            "DCS WR" => {
              let bytes = cfio_bytes(x.item);
              log_to_file(&format!("System DCS WR bytes: {}", bytes));
              rs.sys_wr_bw += bytes;
            }
            _ => {
              log_to_file(&format!("Unhandled AMC Stats channel: {}", x.channel));
            }
          }
        }

        process_bandwidth_histograms(&x.group, &x.subgroup, &x.channel, x.item, &mut rs);
      }

      rs.ecpu_usage = calc_freq_final(&ecpu_usages, &self.soc.ecpu_freqs);
      rs.pcpu_usage = calc_freq_final(&pcpu_usages, &self.soc.pcpu_freqs);
      rs.sys_rd_bw =
        zero_div(rs.sys_rd_bw.saturating_sub(rs.ane_rd_bw) as f64, sample_dt as f64 / 1000.0)
          as u64; // bytes/sec
      rs.sys_wr_bw =
        zero_div(rs.sys_wr_bw.saturating_sub(rs.ane_wr_bw) as f64, sample_dt as f64 / 1000.0)
          as u64; // bytes/sec
      rs.ane_rd_bw = zero_div(rs.ane_rd_bw as f64, sample_dt as f64 / 1000.0) as u64; // bytes/sec
      rs.ane_wr_bw = zero_div(rs.ane_wr_bw as f64, sample_dt as f64 / 1000.0) as u64; // bytes/sec

      results.push(rs);
    }

    let mut rs = Metrics::default();
    rs.ecpu_usage.0 = zero_div(results.iter().map(|x| x.ecpu_usage.0).sum(), measures as _);
    rs.ecpu_usage.1 = zero_div(results.iter().map(|x| x.ecpu_usage.1).sum(), measures as _);
    rs.pcpu_usage.0 = zero_div(results.iter().map(|x| x.pcpu_usage.0).sum(), measures as _);
    rs.pcpu_usage.1 = zero_div(results.iter().map(|x| x.pcpu_usage.1).sum(), measures as _);
    rs.gpu_usage.0 = zero_div(results.iter().map(|x| x.gpu_usage.0).sum(), measures as _);
    rs.gpu_usage.1 = zero_div(results.iter().map(|x| x.gpu_usage.1).sum(), measures as _);
    rs.cpu_power = zero_div(results.iter().map(|x| x.cpu_power).sum(), measures as _);
    rs.gpu_power = zero_div(results.iter().map(|x| x.gpu_power).sum(), measures as _);
    rs.ane_power = zero_div(results.iter().map(|x| x.ane_power).sum(), measures as _);
    rs.all_power = rs.cpu_power + rs.gpu_power + rs.ane_power;

    rs.sys_rd_bw = zero_div(results.iter().map(|x| x.sys_rd_bw).sum(), measures as _); // bytes/sec
    rs.sys_wr_bw = zero_div(results.iter().map(|x| x.sys_wr_bw).sum(), measures as _); // bytes/sec
    rs.ane_rd_bw = zero_div(results.iter().map(|x| x.ane_rd_bw).sum(), measures as _); // bytes/sec
    rs.ane_wr_bw = zero_div(results.iter().map(|x| x.ane_wr_bw).sum(), measures as _); // bytes/sec

    let (
        ane_bw_rd_hist, 
        ane_bw_wr_hist,
        sys_rd_bw_hist, 
        sys_rw_bw_hist, 
        floor_rd_bw_hist, 
        floor_rw_bw_hist
    ) = consolidate_histograms(&results, measures);

    rs.ane_bw_rd_hist = ane_bw_rd_hist;
    rs.ane_bw_wr_hist = ane_bw_wr_hist;
    rs.sys_rd_bw_hist = sys_rd_bw_hist;
    rs.sys_rw_bw_hist = sys_rw_bw_hist;
    rs.floor_rd_bw_hist = floor_rd_bw_hist;
    rs.floor_rw_bw_hist = floor_rw_bw_hist;

    // Calculate and log the average bandwidth
    let ane_avg_rd_bw = rs.ane_bw_rd_hist.weighted_average_gbps("ane_avg_rd_bw");
    let ane_avg_wr_bw = rs.ane_bw_wr_hist.weighted_average_gbps("ane_avg_wr_bw");
    let sys_rd_bw = rs.sys_rd_bw_hist.weighted_average_gbps("sys_rd_bw");
    let sys_rw_bw = rs.sys_rw_bw_hist.weighted_average_gbps("sys_rw_bw");
    let floor_rd_avg_bw = rs.floor_rd_bw_hist.weighted_average_gbps("floor_rd_avg_bw");
    let floor_rw_avg_bw = rs.floor_rw_bw_hist.weighted_average_gbps("floor_rw_avg_bw");
    //if rs.ane_rd_bw == 0 {
    //  rs.ane_rd_bw = (avg_bw * 1024.0 * 1024.0 * 1024.0) as u64; // Convert GB/s to B/s
    //}

    let freq_adj = 1.0 ; // scale- experimental
    
    // Use Bandwidth Histograms if failed to get values from AMC Stats
    // this is required for M4 Macs (Sequia)
    
    if rs.ane_rd_bw==0 {
       rs.ane_rd_bw = (ane_avg_rd_bw * 1024.0 * 1024.0 * 1024.0) as u64; // Convert GB/s to B/s
    }
    
    if rs.ane_wr_bw==0 {
      rs.ane_wr_bw = ((ane_avg_wr_bw - ane_avg_rd_bw) * 1024.0 * 1024.0 * 1024.0) as u64; // Convert GB/s to B/s
    }
    
    if rs.sys_rd_bw==0 {
      rs.sys_rd_bw = ((sys_rd_bw - floor_rd_avg_bw) * 1024.0 * 1024.0 * 1024.0 * freq_adj ) as u64; // Convert GB/s to B/s
    }
    
    if rs.sys_wr_bw==0 {
      rs.sys_wr_bw = (
        ((sys_rw_bw - floor_rw_avg_bw)-(sys_rd_bw - floor_rd_avg_bw))
        * 1024.0 * 1024.0 * 1024.0 * freq_adj 
      ) as u64; // Convert GB/s to B/s
    }

    rs.sys_wr_bw = rs.sys_wr_bw * freq_adj as u64;
    
    log_to_file(&format!("Average ANE bandwidth: {} GB/s", ane_avg_rd_bw));
    log_to_file(&format!("Average System bandwidth: {} GB/s", sys_rd_bw));

    rs.memory = self.get_mem()?;
    rs.temp = self.get_temp()?;

    rs.sys_power = match self.get_sys_power() {
      Ok(val) => val.max(rs.all_power),
      Err(_) => 0.0,
    };

    log_to_file(&format!("Final bandwidth values:"));
    log_to_file(&format!("ANE read: {} B/s", rs.ane_rd_bw));
    log_to_file(&format!("ANE write: {} B/s", rs.ane_wr_bw));
    log_to_file(&format!("System read: {} B/s", rs.sys_rd_bw));
    log_to_file(&format!("System write: {} B/s", rs.sys_wr_bw));

    Ok(rs)
  }
}

fn update_histogram(
    hist: &mut MetricHistogram,
    events: Vec<(String, u64)>,
    channel_name: &str,
) {
    log_to_file(&format!("Found AMCC Stats channel: {}", channel_name));
    if hist.values.len() == 0 {
        hist.labels = events.iter().map(|x| x.0.clone()).collect();
        hist.values = vec![0; events.len()];
    }
    assert!(
        hist.values.len() == events.len(),
        "AMCC BW length mismatch: {} != {}",
        hist.values.len(),
        events.len()
    );
    for i in 0..events.len() {
        hist.values[i] += events[i].1 as u64;
    }
}

fn process_bandwidth_histograms(
    group: &str,
    subgroup: &str,
    channel: &str,
    item: CFDictionaryRef,
    rs: &mut Metrics,
) {
    if group != "PMP" {
        return;
    }

    let events = cfio_get_residencies(item);
    let events: Vec<(String, u64)> = events
        .into_iter()
        .map(|(s, v)| (s, v.max(0) as u64))
        .collect();

    match (subgroup, channel) {
        ("DCS BW", s) if s.starts_with("AMCC") && s.ends_with("RD") => {
            update_histogram(&mut rs.sys_rd_bw_hist, events, channel);
        }
        ("DCS BW", s) if s.starts_with("AMCC") && s.ends_with("RD+WR") => {
            update_histogram(&mut rs.sys_rw_bw_hist, events, channel);
        }
        ("AFR BW", s) if s.starts_with("AMCC") && s.ends_with("RD") => {
            update_histogram(&mut rs.floor_rd_bw_hist, events, channel);
        }
        ("AFR BW", s) if s.starts_with("AMCC") && s.ends_with("RD+WR") => {
            update_histogram(&mut rs.floor_rw_bw_hist, events, channel);
        }
        ("AF BW", s) if s.starts_with("ANE") && s.ends_with("RD") => {
            update_histogram(&mut rs.ane_bw_rd_hist, events, channel);
        }
        ("AF BW", s) if s.starts_with("ANE") && s.ends_with("RD+WR") => {
          update_histogram(&mut rs.ane_bw_wr_hist, events, channel);
      }
        _ => {}
    }
}

fn consolidate_histograms(results: &Vec<Metrics>, measures: usize) -> (
  MetricHistogram, 
  MetricHistogram, 
  MetricHistogram, 
  MetricHistogram, 
  MetricHistogram, 
  MetricHistogram
) {
    let process_histogram = |selector: fn(&Metrics) -> MetricHistogram| {
        results
            .iter()
            .map(|x| selector(x))
            .reduce(|a, b| sum_hist(&a, &b))
            .map(|hist| zero_div_hist(&hist, measures as _))
            .unwrap_or_default()
    };

    (
      
        process_histogram(|x| x.ane_bw_rd_hist.clone()),
        process_histogram(|x| x.ane_bw_wr_hist.clone()),
        process_histogram(|x| x.sys_rd_bw_hist.clone()),
        process_histogram(|x| x.sys_rw_bw_hist.clone()),
        process_histogram(|x| x.floor_rd_bw_hist.clone()),
        process_histogram(|x| x.floor_rw_bw_hist.clone()),
    )
}


fn log_to_file(message: &str) {
  if false {  // disable logging
    let log_path = "macmon_debug.log";
    match OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path) 
    {
        
        Ok(mut file) => {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            
            if let Err(e) = writeln!(file, "[{}] {}", timestamp, message) {
                eprintln!("Failed to write to log file {}: {}", log_path, e);
            }
        }
        Err(e) => {
            eprintln!("Failed to open log file {}: {}", log_path, e);
        }
    }
  }
}