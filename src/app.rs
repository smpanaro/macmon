use std::iter::zip;
use std::{io::stdout, time::Instant};
use std::{sync::mpsc, time::Duration};

use ratatui::crossterm::{
  event::{self, KeyCode, KeyModifiers},
  terminal, ExecutableCommand,
};
use ratatui::{prelude::*, widgets::*};

use crate::config::{Config, ViewType};
use crate::metrics::{zero_div, MetricHistogram, Metrics, Sampler};
use crate::{
  metrics::{MemMetrics, TempMetrics},
  sources::SocInfo,
};

type WithError<T> = Result<T, Box<dyn std::error::Error>>;

const GB: u64 = 1024 * 1024 * 1024;
const MAX_SPARKLINE: usize = 128;

// MARK: Term utils

fn enter_term() -> Terminal<impl Backend> {
  std::panic::set_hook(Box::new(|info| {
    leave_term();
    eprintln!("{}", info);
  }));

  terminal::enable_raw_mode().unwrap();
  stdout().execute(terminal::EnterAlternateScreen).unwrap();

  let term = CrosstermBackend::new(std::io::stdout());
  let term = Terminal::new(term).unwrap();
  term
}

fn leave_term() {
  terminal::disable_raw_mode().unwrap();
  stdout().execute(terminal::LeaveAlternateScreen).unwrap();
}

// MARK: Storage

fn items_add<T>(vec: &mut Vec<T>, val: T) -> &Vec<T> {
  vec.insert(0, val);
  if vec.len() > MAX_SPARKLINE {
    vec.pop();
  }
  vec
}

#[derive(Debug, Default)]
struct FreqStore {
  items: Vec<u64>, // from 0 to 100
  top_value: u64,
  usage: f64, // from 0.0 to 1.0
}

impl FreqStore {
  fn push(&mut self, value: u64, usage: f64) {
    items_add(&mut self.items, (usage * 100.0) as u64);
    self.top_value = value;
    self.usage = usage;
  }
}

#[derive(Debug, Default)]
struct PowerStore {
  items: Vec<u64>,
  top_value: f64,
  max_value: f64,
  avg_value: f64,
}

impl PowerStore {
  fn push(&mut self, value: f64) {
    items_add(&mut self.items, (value * 1000.0) as u64);
    self.top_value = value;
    self.avg_value = self.items.iter().sum::<u64>() as f64 / self.items.len() as f64 / 1000.0;
    self.max_value = self.items.iter().max().map_or(0, |v| *v) as f64 / 1000.0;
  }
}

#[derive(Debug, Default)]
struct MemoryStore {
  items: Vec<u64>,
  ram_usage: u64,
  ram_total: u64,
  swap_usage: u64,
  swap_total: u64,
  max_ram: u64,
}

impl MemoryStore {
  fn push(&mut self, value: MemMetrics) {
    items_add(&mut self.items, value.ram_usage);
    self.ram_usage = value.ram_usage;
    self.ram_total = value.ram_total;
    self.swap_usage = value.swap_usage;
    self.swap_total = value.swap_total;
    self.max_ram = self.items.iter().max().map_or(0, |v| *v);
  }
}

#[derive(Debug, Default)]
struct BandwidthStore {
  items: Vec<u64>,
  current: f64,
  max: f64,
  avg: f64,
}

impl BandwidthStore {
  fn push(&mut self, value: u64) {
    items_add(&mut self.items, value);
    self.current = value as f64;
    self.max = self.max.max(value as f64);
    self.avg = (self.items.iter().sum::<u64>() as f64 / self.items.len() as f64).round();
  }
}

// MARK: Components

fn h_stack(area: Rect) -> (Rect, Rect) {
  let ha = Layout::default()
    .direction(Direction::Horizontal)
    .constraints([Constraint::Fill(1), Constraint::Fill(1)].as_ref())
    .split(area);

  (ha[0], ha[1])
}

fn v_stack(area: Rect) -> (Rect, Rect) {
  let ha = Layout::default()
    .direction(Direction::Vertical)
    .constraints([Constraint::Fill(1), Constraint::Fill(1)].as_ref())
    .split(area);

  (ha[0], ha[1])
}

// MARK: Threads

enum Event {
  Update(Metrics),
  ChangeColor,
  ChangeView,
  Tick,
  Quit,
}

fn handle_key_event(key: &event::KeyEvent, tx: &mpsc::Sender<Event>) -> WithError<()> {
  match key.code {
    KeyCode::Char('q') => Ok(tx.send(Event::Quit)?),
    KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => Ok(tx.send(Event::Quit)?),
    KeyCode::Char('c') => Ok(tx.send(Event::ChangeColor)?),
    KeyCode::Char('v') => Ok(tx.send(Event::ChangeView)?),
    _ => Ok(()),
  }
}

fn run_inputs_thread(tx: mpsc::Sender<Event>, tick: u64) {
  let tick_rate = Duration::from_millis(tick);

  std::thread::spawn(move || {
    let mut last_tick = Instant::now();

    loop {
      if event::poll(Duration::from_millis(tick)).unwrap() {
        match event::read().unwrap() {
          event::Event::Key(key) => handle_key_event(&key, &tx).unwrap(),
          _ => {}
        };
      }

      if last_tick.elapsed() >= tick_rate {
        tx.send(Event::Tick).unwrap();
        last_tick = Instant::now();
      }
    }
  });
}

fn run_sampler_thread(tx: mpsc::Sender<Event>, interval: u64) {
  let interval = interval.max(100).min(10000);

  std::thread::spawn(move || {
    let mut sampler = Sampler::new().unwrap();

    // Send initial metrics
    tx.send(Event::Update(sampler.get_metrics(100).unwrap())).unwrap();

    loop {
      tx.send(Event::Update(sampler.get_metrics(interval).unwrap())).unwrap();
    }
  });
}

// MARK: App

#[derive(Debug, Default)]
pub struct App {
  cfg: Config,

  soc: SocInfo,
  mem: MemoryStore,
  temp: TempMetrics,

  cpu_power: PowerStore,
  gpu_power: PowerStore,
  ane_power: PowerStore,
  all_power: PowerStore,
  sys_power: PowerStore,

  ecpu_freq: FreqStore,
  pcpu_freq: FreqStore,
  igpu_freq: FreqStore,

  ane_rd_bw: BandwidthStore,
  ane_wr_bw: BandwidthStore,
  ane_total_bw: BandwidthStore,
  sys_rd_bw: BandwidthStore,
  sys_wr_bw: BandwidthStore,
  sys_total_bw: BandwidthStore,
  all_rd_bw: BandwidthStore,
  all_wr_bw: BandwidthStore,
  all_total_bw: BandwidthStore,
  ane_bw_rd_hist: MetricHistogram,
}

impl App {
  pub fn new() -> WithError<Self> {
    let soc = SocInfo::new()?;
    let cfg = Config::load();
    Ok(Self { cfg, soc, ..Default::default() })
  }

  fn update_metrics(&mut self, data: Metrics) {
    self.cpu_power.push(data.cpu_power as f64);
    self.gpu_power.push(data.gpu_power as f64);
    self.ane_power.push(data.ane_power as f64);
    self.all_power.push(data.all_power as f64);
    self.sys_power.push(data.sys_power as f64);
    self.ecpu_freq.push(data.ecpu_usage.0 as u64, data.ecpu_usage.1 as f64);
    self.pcpu_freq.push(data.pcpu_usage.0 as u64, data.pcpu_usage.1 as f64);
    self.igpu_freq.push(data.gpu_usage.0 as u64, data.gpu_usage.1 as f64);
    self.temp = data.temp;
    self.mem.push(data.memory);
    self.ane_rd_bw.push(data.ane_rd_bw);
    self.ane_wr_bw.push(data.ane_wr_bw);
    self.ane_total_bw.push(data.ane_rd_bw + data.ane_wr_bw);
    self.sys_rd_bw.push(data.sys_rd_bw);
    self.sys_wr_bw.push(data.sys_wr_bw);
    self.sys_total_bw.push(data.sys_rd_bw + data.sys_wr_bw);
    self.all_rd_bw.push(data.ane_rd_bw + data.sys_rd_bw);
    self.all_wr_bw.push(data.ane_wr_bw + data.sys_wr_bw);
    self.all_total_bw.push(data.ane_rd_bw + data.ane_wr_bw + data.sys_rd_bw + data.sys_wr_bw);
    self.ane_bw_rd_hist = data.ane_bw_rd_hist;
  }

  fn title_block<'a>(&self, label_l: &str, label_r: &str) -> Block<'a> {
    let mut block = Block::new()
      .borders(Borders::ALL)
      .border_type(BorderType::Rounded)
      .border_style(self.cfg.color)
      // .title_style(Style::default().gray())
      .padding(Padding::ZERO);

    if label_l.len() > 0 {
      block = block.title_top(Line::from(format!(" {label_l} ")));
    }

    if label_r.len() > 0 {
      block = block.title_top(Line::from(format!(" {label_r} ")).alignment(Alignment::Right));
    }

    block
  }

  fn borderless_title<'a>(&self, label_l: &str, label_r: &str) -> Block<'a> {
    let mut block = Block::new()
      .borders(Borders::TOP)
      .border_type(BorderType::Double)
      .title_style(self.cfg.color)
      .border_style(self.cfg.color)
      .padding(Padding::ZERO);

    if label_l.len() > 0 {
      block = block.title_top(Line::from(format!(" {label_l} ")));
    }

    if label_r.len() > 0 {
      block = block.title_top(Line::from(format!(" {label_r} ")).alignment(Alignment::Right));
    }

    block
  }

  fn get_power_block<'a>(&self, label: &str, val: &'a PowerStore, temp: f32) -> Sparkline<'a> {
    let label_l = format!(
      "{} {:.2}W ({:.2}, {:.2})",
      // "{} {:.2}W (avg: {:.2}W, max: {:.2}W)",
      // "{} {:.2}W (~{:.2}W ^{:.2}W)",
      label,
      val.top_value,
      val.avg_value,
      val.max_value
    );

    let label_r = if temp > 0.0 { format!("{:.1}°C", temp) } else { "".to_string() };

    Sparkline::default()
      .block(self.title_block(label_l.as_str(), label_r.as_str()))
      .direction(RenderDirection::RightToLeft)
      .data(&val.items)
      .style(self.cfg.color)
  }

  fn get_bandwidth_block<'a>(&self, label: &str, val: &'a BandwidthStore) -> Sparkline<'a> {
    let label = format!(
      "{} {:.2} GB/s ({:.2}, {:.2})",
      label,
      val.current / 1e9,
      val.avg / 1e9,
      val.max / 1e9
    );

    Sparkline::default()
      .block(self.title_block(&label, ""))
      .direction(RenderDirection::RightToLeft)
      .data(val.items.iter().map(|&x| x as u64).collect::<Vec<u64>>())
      .max(val.max as u64)
      .style(self.cfg.color)
  }

  fn get_bandwidth_hist_block<'a>(&self, label: &str, val: &'a MetricHistogram) -> BarChart<'a> {
    let bars: Vec<Bar> = zip(val.labels.iter(), val.values.iter())
      .enumerate()
      .map(|(i, (label, value))| {
        Bar::default()
          .value(*value)
          .label(match i % 2 {
            0 => Line::from(format!("{}", label.trim().replace("GB/s", ""))),
            _ => Line::default(),
          })
          .text_value(String::new())
          .style(self.cfg.color)
      })
      .collect();

    BarChart::default()
      .block(self.title_block(label, ""))
      .bar_width(2)
      .bar_gap(0)
      .data(BarGroup::default().bars(&bars))
      .style(self.cfg.color)
  }

  fn render_freq_block(&self, f: &mut Frame, r: Rect, label: &str, val: &FreqStore) {
    let label = format!("{} {:3.0}% @ {:4.0} MHz", label, val.usage * 100.0, val.top_value);
    let block = self.title_block(label.as_str(), "");

    match self.cfg.view_type {
      ViewType::Sparkline => {
        let w = Sparkline::default()
          .block(block)
          .direction(RenderDirection::RightToLeft)
          .data(&val.items)
          .max(100)
          .style(self.cfg.color);
        f.render_widget(w, r);
      }
      ViewType::Gauge => {
        let w = Gauge::default()
          .block(block)
          .gauge_style(self.cfg.color)
          .style(self.cfg.color)
          .label("")
          .ratio(val.usage);
        f.render_widget(w, r);
      }
    }
  }

  fn render_mem_block(&self, f: &mut Frame, r: Rect, val: &MemoryStore) {
    let ram_usage_gb = val.ram_usage as f64 / GB as f64;
    let ram_total_gb = val.ram_total as f64 / GB as f64;

    let swap_usage_gb = val.swap_usage as f64 / GB as f64;
    let swap_total_gb = val.swap_total as f64 / GB as f64;

    let label_l = format!("RAM {:4.2} / {:4.1} GB", ram_usage_gb, ram_total_gb);
    let label_r = format!("SWAP {:.2} / {:.1} GB", swap_usage_gb, swap_total_gb);

    let block = self.title_block(label_l.as_str(), label_r.as_str());
    match self.cfg.view_type {
      ViewType::Sparkline => {
        let w = Sparkline::default()
          .block(block)
          .direction(RenderDirection::RightToLeft)
          .data(&val.items)
          .max(val.ram_total)
          .style(self.cfg.color);
        f.render_widget(w, r);
      }
      ViewType::Gauge => {
        let w = Gauge::default()
          .block(block)
          .gauge_style(self.cfg.color)
          .style(self.cfg.color)
          .label("")
          .ratio(zero_div(ram_usage_gb, ram_total_gb));
        f.render_widget(w, r);
      }
    }
  }

  fn render(&mut self, f: &mut Frame) {
    let label_l = format!(
      "{} ({}E+{}P+{}GPU {}GB)",
      self.soc.chip_name,
      self.soc.ecpu_cores,
      self.soc.pcpu_cores,
      self.soc.gpu_cores,
      self.soc.memory_gb,
    );

    let rows = Layout::default()
      .direction(Direction::Vertical)
      .constraints([Constraint::Fill(3), Constraint::Fill(2), Constraint::Fill(2)].as_ref())
      .split(f.area());

    let brand = format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
    let block = self.title_block(&label_l, &brand);
    let iarea = block.inner(rows[0]);
    f.render_widget(block, rows[0]);

    let iarea = Layout::default()
      .direction(Direction::Vertical)
      .constraints([Constraint::Fill(1), Constraint::Fill(1)].as_ref())
      .split(iarea);

    // 1st row
    let (c1, c2) = h_stack(iarea[0]);
    self.render_freq_block(f, c1, "E-CPU", &self.ecpu_freq);
    self.render_freq_block(f, c2, "P-CPU", &self.pcpu_freq);

    // 2nd row
    let (c1, c2) = h_stack(iarea[1]);
    self.render_mem_block(f, c1, &self.mem);
    self.render_freq_block(f, c2, "GPU", &self.igpu_freq);

    // Bandwidth row
    let bw_label = format!(
      "Bandwidth: {:.2} GB/s (avg. {:.2} GB/s, max. {:.2} GB/s)",
      self.all_total_bw.current / 1e9,
      self.all_total_bw.avg / 1e9,
      self.all_total_bw.max / 1e9
    );
    let bw_block = self.title_block(&bw_label, "");
    let bw_area = bw_block.inner(rows[1]);
    f.render_widget(bw_block, rows[1]);

    let bw_area = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([Constraint::Fill(1), Constraint::Fill(1)].as_ref())
      .split(bw_area);

    // 1st column
    let sys_bw_label = format!(
      "System {:.2} GB/s ({:.2}, {:.2})",
      self.sys_total_bw.current / 1e9,
      self.sys_total_bw.avg / 1e9,
      self.sys_total_bw.max / 1e9
    );
    let sys_bw = self.borderless_title(&sys_bw_label, "");
    let iarea = sys_bw.inner(bw_area[0]);
    f.render_widget(sys_bw, bw_area[0]);
    let (c1, c2) = v_stack(iarea);
    f.render_widget(self.get_bandwidth_block("Read", &self.sys_rd_bw), c1);
    f.render_widget(self.get_bandwidth_block("Write", &self.sys_wr_bw), c2);

    // 2nd column
    let ane_bw_label = format!(
      "ANE {:.2} GB/s ({:.2}, {:.2})",
      self.ane_total_bw.current / 1e9,
      self.ane_total_bw.avg / 1e9,
      self.ane_total_bw.max / 1e9,
    );
    if self.cfg.view_type == ViewType::Gauge {
      // Bars in this case are basically many gauges.
      f.render_widget(self.get_bandwidth_hist_block(&ane_bw_label, &self.ane_bw_rd_hist), bw_area[1]);
    } else {
      let ane_bw = self.borderless_title(&ane_bw_label, "");
      let iarea = ane_bw.inner(bw_area[1]);
      f.render_widget(ane_bw, bw_area[1]);
      let (c1, c2) = v_stack(iarea);
      f.render_widget(self.get_bandwidth_block("Read", &self.ane_rd_bw), c1);
      f.render_widget(self.get_bandwidth_block("Write", &self.ane_wr_bw), c2);
    }

    // Power row
    let label_l = format!(
      "Power: {:.2}W (avg {:.2}W, max {:.2}W)",
      self.all_power.top_value, self.all_power.avg_value, self.all_power.max_value,
    );

    // Show label only if sensor is available
    let label_r = if self.sys_power.top_value > 0.0 {
      format!(
        "Total {:.2}W ({:.2}, {:.2})",
        self.sys_power.top_value, self.sys_power.avg_value, self.sys_power.max_value
      )
    } else {
      "".to_string()
    };

    let block = self.title_block(&label_l, &label_r);
    let usage = " Press 'q' to quit, 'c' – color, 'v' – view ";
    let block = block.title_bottom(Line::from(usage).right_aligned());
    let iarea = block.inner(rows[2]);
    f.render_widget(block, rows[2]);

    let ha = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([Constraint::Fill(1), Constraint::Fill(1), Constraint::Fill(1)].as_ref())
      .split(iarea);

    f.render_widget(self.get_power_block("CPU", &self.cpu_power, self.temp.cpu_temp_avg), ha[0]);
    f.render_widget(self.get_power_block("GPU", &self.gpu_power, self.temp.gpu_temp_avg), ha[1]);
    f.render_widget(self.get_power_block("ANE", &self.ane_power, 0.0), ha[2]);
  }

  pub fn run_loop(&mut self, interval: u64) -> WithError<()> {
    let (tx, rx) = mpsc::channel::<Event>();
    run_inputs_thread(tx.clone(), 250);
    run_sampler_thread(tx.clone(), interval);

    let mut term = enter_term();

    loop {
      term.draw(|f| self.render(f)).unwrap();

      match rx.recv()? {
        Event::Quit => break,
        Event::Update(data) => self.update_metrics(data),
        Event::ChangeColor => self.cfg.next_color(),
        Event::ChangeView => self.cfg.next_view_type(),
        _ => {}
      }
    }

    leave_term();
    Ok(())
  }
}
