import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numba import jit
from scipy.integrate import odeint
from scipy import optimize
from collections import namedtuple
from IPython.display import clear_output
import matplotlib.dates as mdates
from datetime import datetime
DATES_FORMATTER = mdates.DateFormatter('%m-%d')
TODAY = mdates.date2num(datetime.today())
countries = [
 'US', 'Spain', 'Italy', 'France', 'Germany', 'United Kingdom', 'China', 'Iran', 'Turkey', 'Belgium', 'Netherlands', 'Canada', 'Switzerland', 'Brazil', 'Russia', 'Portugal', 'Austria', 'Israel', 'Ireland', 'Sweden', 'India', 'Korea, South']
covid_db_names = {'USA':'US', 
 'Korea':'Korea, South', 
 'UK':'United Kingdom'}
population_db_names = {'US':'United States', 
 'USA':'United States', 
 'Russia':'Russian Federation', 
 'UK':'United Kingdom', 
 'Iran':'Iran, Islamic Rep.', 
 'Korea':'Korea, Rep.'}
_data = None

class CovidData:

    def __init__(self):
        self.loaded = False
        self.covid_confirmed = None
        self.covid_deaths = None
        self.covid_recovered = None

    def _ensure(self, reload=False):
        global _data
        if not self.loaded:
            if reload or _data is None:
                _data = namedtuple('covid_data', ('covid_confirmed', 'covid_deaths',
                                                  'covid_recovered'))
                COVID_CONFIRMED_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
                COVID_DEATHS_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
                COVID_RECOVERED_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
                _data.covid_confirmed = pd.read_csv(COVID_CONFIRMED_URL)
                _data.covid_deaths = pd.read_csv(COVID_DEATHS_URL)
                _data.covid_recovered = pd.read_csv(COVID_RECOVERED_URL)
                _data.covid_confirmed.rename(columns={'Country/Region': 'Country'}, inplace=True)
                _data.covid_deaths.rename(columns={'Country/Region': 'Country'}, inplace=True)
                _data.covid_recovered.rename(columns={'Country/Region': 'Country'}, inplace=True)
                _data.covid_confirmed.rename(columns={'Province/State': 'State'}, inplace=True)
                _data.covid_deaths.rename(columns={'Province/State': 'State'}, inplace=True)
                _data.covid_recovered.rename(columns={'Province/State': 'State'}, inplace=True)
                _data.covid_confirmed = _data.covid_confirmed.groupby('Country').sum()
                _data.covid_deaths = _data.covid_deaths.groupby('Country').sum()
                _data.covid_recovered = _data.covid_recovered.groupby('Country').sum()
            self.covid_confirmed = _data.covid_confirmed
            self.covid_deaths = _data.covid_deaths
            self.covid_recovered = _data.covid_recovered
            self.loaded = True
        return self

    def countries(self):
        return self._ensure().covid_confirmed.index

    def __getitem__(self, key):
        self._ensure()
        _confirmed = self.covid_confirmed.loc[key].values[2:].astype(int)
        data = np.empty((3, _confirmed.shape[0]), dtype=(np.int64))
        data[0] = _confirmed
        data[1] = self.covid_deaths.loc[key].values[2:].astype(int)
        data[2] = self.covid_recovered.loc[key].values[2:].astype(int)
        return data


class PopulationData:

    def __init__(self):
        self.loaded = False
        self.population = None

    def _ensure(self, reload=False):
        if not self.loaded or reload:
            population = pd.read_csv('population.csv')
            population.rename(columns={'Country Name': 'Country'}, inplace=True)
            self.population = population[['Country', '2018']].set_index(['Country'])['2018']
            self.loaded = True
        return self

    def __getitem__(self, key):
        return self._ensure().population[key]

    def countries(self):
        return self._ensure().population.index


ComputationalModel = namedtuple('ComputationalModel', ('name', 'rhs_fn', 'iv_fn', 'param_count',
                                                       'bounds', 'solution_spec',
                                                       'solution_columns', 'report_columns',
                                                       'population_columns'))
REVEAL_ESTIMATE = 0.005

@jit
def sis_o_rhs(x, t, params):
    S, I, X = x
    a, i_s, k = params
    dsdt = (-a * S + i_s) * I
    didt = (a * S - i_s) * I
    dxdt = k * I
    return np.array([dsdt, didt, dxdt])


def mk_sis_o_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='sis_o', rhs_fn=sis_o_rhs,
      iv_fn=(lambda v0: (
     1 - v0[0], v0[0], model.cases()[0])),
      param_count=3,
      bounds=(
     np.array([0.0, 0.0, model.cases()[0]]),
     np.array([5.0, 4.0, 0.0001])),
      solution_spec=('S', 'I', 'O'),
      solution_columns=('Susceptible', 'Infected', 'Observed'),
      report_columns=('alpha', 'beta', 'reveal efficiency', 'Infected initial'),
      population_columns=('Infected initial', ))


@jit
def sir_o_rhs(x, t, params):
    S, I, R, X = x
    a, b, k = params
    dsdt = -a * S * I
    didt = (a * S - b) * I
    dxdt = k * I
    dRdt = b * I
    return np.array([dsdt, didt, dRdt, dxdt])


def mk_sir_o_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='sir_o', rhs_fn=sir_o_rhs,
      iv_fn=(lambda v0: (
     1 - v0[0], v0[0], 0.0, model.cases()[0])),
      param_count=3,
      bounds=(
     np.array([0.0, 0.0, 0.0, model.cases()[0]]),
     np.array([5.0, 4.0, 0.3, limit])),
      solution_spec=('S', 'I', 'R', 'O'),
      solution_columns=('Susceptible', 'Infected', 'Recovered', 'Observed'),
      report_columns=('alpha', 'beta', 'reveal efficiency', 'Infected initial'),
      population_columns=('Infected initial', ))


def mk_sir_o_r0_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='sir_o_r0', rhs_fn=sir_o_rhs,
      iv_fn=(lambda v0: (
     1 - v0[0], v0[0], v0[1], model.cases()[0])),
      param_count=3,
      bounds=(
     np.array([0.0, 0.0, 0.0, model.cases()[0], 0.0]),
     np.array([5.0, 4.0, 0.3, limit, limit])),
      solution_spec=('S', 'I', 'R', 'O'),
      solution_columns=('Susceptible', 'Infected', 'Recovered', 'Observed'),
      report_columns=('alpha', 'beta', 'reveal efficiency', 'Infected initial', 'Recovered initial'),
      population_columns=('Infected initial', 'Recovered initial'))


@jit
def sir_so_rhs(x, t, params):
    S, I, R, X = x
    a, b, k, r_s = params
    dsdt = -a * S * I + r_s * R
    didt = (a * S - b) * I
    dxdt = k * I
    dRdt = b * I - r_s * R
    return np.array([dsdt, didt, dRdt, dxdt])


def mk_sir_so_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='sir_so', rhs_fn=sir_so_rhs,
      iv_fn=(lambda v0: (
     1 - v0[0], v0[0], 0.0, model.cases()[0])),
      param_count=4,
      bounds=(
     np.array([0.0, 0.0, 0.0, 0.0, model.cases()[0]]),
     np.array([5.0, 4.0, 3.0, 0.3, limit])),
      solution_spec=('S', 'I', 'R', 'O'),
      solution_columns=('Susceptible', 'Infected', 'Recovered', 'Observed'),
      report_columns=('alpha', 'beta', 'recovered to susceptible', 'reveal efficiency',
                      'Infected initial'),
      population_columns=('Infected initial', ))


def mk_sir_so_r0_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='sir_so_r0', rhs_fn=sir_so_rhs,
      iv_fn=(lambda v0: (
     1 - v0[0] - v0[1], v0[0], v0[1], model.cases()[0])),
      param_count=4,
      bounds=(
     np.array([0.0, 0.0, 0.0, 0.0, model.cases()[0], 0.0]),
     np.array([5.0, 4.0, 3.0, 0.3, limit, limit])),
      solution_spec=('S', 'I', 'R', 'O'),
      solution_columns=('Susceptible', 'Infected', 'Recovered', 'Observed'),
      report_columns=('alpha', 'beta', 'recovered to susceptible', 'reveal efficiency',
                      'Infected initial', 'Recovered initial'),
      population_columns=('Infected initial', 'Recovered initial'))


@jit
def sir_co_rhs(x, t, params):
    S, I, R, C, O = x
    a, b, a_c, i_r, c_r, c_i, k = params
    dsdt = -a * (I + a_c * C) * S
    didt = a * (I + a_c * C) * S - b * I + c_i * C
    dRdt = b * i_r * I + c_r * C
    dCdt = b * (1 - i_r) * I - (c_r + c_i)* C
    dxdt = k * (I + C)
    return np.array([dsdt, didt, dRdt, dCdt, dxdt])


def mk_sir_co_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='sir_co', rhs_fn=sir_co_rhs,
      iv_fn=(lambda v0: (1 - v0[0] - v0[1] - v0[2], v0[0], v0[1], v0[2], model.cases()[0])),
      param_count=7,
      bounds=(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, model.cases()[0], 0.0, 0.0]),
              np.array([5.0, 4.0, 1.0, 1.0, 3.0, 3.0, 0.05, limit, limit, limit])),
      solution_spec=[
     'S', 'I', 'R', 'C', 'O'],
      solution_columns=[
     'Succeptible', 'Infectious', 'Recovered', 'Carrier', 'Observed'],
      report_columns=('alpha', 'beta', 'carrier efficiency', 'infectious to recovered',
                      'carrier recovery', 'carrier to infectious', 'reveal efficiency', 
                      'Initial infected', 'Initial carrier', 'Initial recovered'),
      population_columns=('Initial infected', 'Initial carrier', 'Initial recovered'))


@jit
def sir_sco_rhs(x, t, params):
    S, I, R, C, X = x
    a, b, a_c, i_r, c_r, c_i, r_s, k = params
    dsdt = -a * (I + a_c * C) * S + r_s * R
    didt = a * (I + a_c * C) * S - b * I + c_i*C
    dRdt = b * i_r * I + c_r * C - r_s * R
    dCdt = b * (1 - i_r) * I - (c_r +c_i)*C
    dxdt = k * (I + C)
    return np.array([dsdt, didt, dRdt, dCdt, dxdt])


def mk_sir_sco_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='sir_sco', rhs_fn=sir_sco_rhs,
      iv_fn=(lambda v0: (1 - v0[0] - v0[1] - v0[2], v0[0], v0[1], v0[2], model.cases()[0])),
      param_count=8,
      bounds=(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, model.cases()[0], 0.0, 0.0]),
              np.array([5.0, 4.0, 1.0, 1.0, 3.0, 3.0, 3.0, 0.05, limit, limit, limit])),
      solution_spec=[
     'S', 'I', 'R', 'C', 'O'],
      solution_columns=[
     'Succeptible', 'Infectious', 'Recovered', 'Carrier', 'Observed'],
      report_columns=('alpha', 'beta', 'carrier efficiency', 'infectious to recovered',
                      'carrier recovery', 'carrier to infectious', 'recovered to susceptible', 'reveal efficiency',
                      'Initial infected', 'Initial carrier', 'Initial recovered'),
      population_columns=('Initial infected', 'Initial carrier', 'Initial recovered'))


@jit
def sir_sco_kc_rhs(x, t, params):
    S, I, R, C, X = x
    a, b, a_c, i_r, c_r, c_i, r_s, k_i, k_c = params
    dsdt = -a * (I + a_c * C) * S + r_s * R
    didt = a * (I + a_c * C) * S - b * I + c_i*C
    dRdt = b * i_r * I + c_r * C - r_s * R
    dCdt = b * (1 - i_r) * I - (c_r + c_i)*C
    dxdt = k_i * I + k_c * C
    return np.array([dsdt, didt, dRdt, dCdt, dxdt])


def mk_sir_sco_kc_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='sir_sco_kc', rhs_fn=sir_sco_kc_rhs,
      iv_fn=(lambda v0: (
     1 - v0[0] - v0[1] - v0[2], v0[0], v0[1], v0[2], model.cases()[0])),
      param_count=9,
      bounds=(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, model.cases()[0], 0.0, 0.0]),
              np.array([5.0, 4.0, 1.0, 1.0, 3.0, 3.0, 3.0, 0.05, 0.05, limit, limit, limit])),
      solution_spec=[
     'S', 'I', 'R', 'C', 'O'],
      solution_columns=[
     'Succeptible', 'Infectious', 'Recovered', 'Carrier', 'Observed'],
      report_columns=('alpha', 'beta', 'carrier efficiency', 'infectious to recovered',
                      'carrier recovery', 'carrier to infectious', 'recovered to susceptible', 'Infectious reveal efficiency',
                      'Carrier reveal efficiency', 'Initial infected', 'Initial carrier',
                      'Initial recovered'),
      population_columns=('Initial infected', 'Initial carrier', 'Initial recovered'))


@jit
def seir_o_rhs(x, t, params):
    S, E, I, R, O = x
    a, b, ei, er, k = params
    dsdt = -a * S * I
    dedt = a * S * I - (ei + er) * E
    didt = ei * E - b * I
    dOdt = k * I
    dRdt = er * E + b * I
    return np.array([dsdt, dedt, didt, dRdt, dOdt])


def mk_seir_o_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='seir_o', rhs_fn=seir_o_rhs,
      iv_fn=(lambda v0: (1 - v0[0] - v0[1] - v0[2], v0[1], v0[0], v0[2], model.cases()[0])),
      param_count=5,
      bounds=(np.array([0.1, 0.01, 0.0, 0.0, 0.0, model.cases()[0], 0.0, 0.0]),
              np.array([5.0, 4.0, 3.0, 3.0, 0.05, limit, limit, limit])),
      solution_spec=('S', 'E', 'I', 'R', 'O'),
      solution_columns=('Susceptible', 'Exposed', 'Infected', 'Recovered', 'Observed'),
      report_columns=('alpha', 'beta', 'exposed to infected', 'exposed to recovered',
                      'reveal efficiency', 'Infected initial', 'Exposed initial',
                      'Recovered initial'),
      population_columns=('Infected initial', 'Exposed initial', 'Recovered initial'))


@jit
def seir_so_rhs(x, t, params):
    S, E, I, R, O = x
    a, b, ei, er, rs, k = params
    dsdt = -a * S * I + rs * R
    dedt = a * S * I - (ei + er) * E
    didt = ei * E - b * I
    dRdt = er * E + b * I - rs * R
    dOdt = k * I
    return np.array([dsdt, dedt, didt, dRdt, dOdt])


def mk_seir_so_model(model):
    limit = model.cases()[0] / REVEAL_ESTIMATE
    return ComputationalModel(name='seir_so', rhs_fn=seir_so_rhs,
      iv_fn=(lambda v0: (1 - v0[0] - v0[1] - v0[2], v0[1], v0[0], v0[2], model.cases()[0])),
      param_count=6,
      bounds=(np.array([0.1, 0.01, 0.0, 0.0, 0.0, 0.0, model.cases()[0], 0.0, 0.0]),
              np.array([5.0, 4.0, 3.0, 3.0, 3.0, 0.05, limit, limit, limit])),
      solution_spec=('S', 'E', 'I', 'R', 'O'),
      solution_columns=('Susceptible', 'Exposed', 'Infected', 'Recovered', 'Observed'),
      report_columns=('alpha', 'beta', 'exposed to infected', 'exposed to recovered',
                      'recovered to susceptible', 'reveal efficiency', 'Infected initial',
                      'Exposed initial', 'Recovered initial'),
      population_columns=('Infected initial', 'Exposed initial', 'Recovered initial'))


sym, fn = (None, None)
models = {}
for sym, fn in globals().items():
    if len(sym) < 10:
        pass
    else:
        if sym[:3] == 'mk_' and sym[-6:] == '_model':
            models[sym[3:-6]] = fn

def integrate(rhs_fn, params, init_vals, dt, N):
    t_max = N * dt
    t = np.linspace(0, t_max, N + 1)
    sol = odeint(rhs_fn, init_vals, t, args=(params,))
    return sol


def update_progress(progress, message=''):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait=True)
    text = 'Progress: [{0}] {1:.1f}% {2}'.format('#' * block + '-' * (bar_length - block), progress * 100, message)
    print(text)


def nop_update_progress(*args, **kwargs):
    pass


INIT_CASES = 10

class CovidModel:

    def __init__(self, country, population_db_country_name=None, population=None, model='sir_so'):
        self.country = country
        if country in covid_db_names:
            country = covid_db_names[country]
        self.data = CovidData()[country]
        if population is None:
            if population_db_country_name is None:
                if self.country in population_db_names:
                    population_db_country_name = population_db_names[self.country]
                else:
                    population_db_country_name = self.country
            self.population = PopulationData()[population_db_country_name]
        else:
            self.population = population
        self.normed_data = self.data / self.population
        self.true_cases = self.normed_data[0]
        self.set_init_cases(INIT_CASES)
        self.steps_per_day = 1
        self.model = None
        self.set_model(model)

    def set_init_cases(self, init_cases):
        self.init_offset = np.argmax(self.data[0] > init_cases)
        self.n_sqrt = np.sqrt(len(self.data[0]) - self.init_offset)
        return self

    def set_model(self, model):
        if callable(model):
            self.model = model(self)
        else:
            model = model.lower()
            if self.model is not None:
                if self.model.name == model:
                    return
            if model in models:
                mk_fn = models[model]
                self.model = mk_fn(self)
            else:
                raise ValueError('Unknown model: ' + model)
        self.random_search_results = []
        self.best = []
        self._best_solutions = {}

    def cases(self):
        return self.true_cases[self.init_offset:]

    def ncases(self):
        return len(self.true_cases) - self.init_offset

    def obs_all(self, params_and_v0, num_days):
        dt = 1.0 / self.steps_per_day
        num = (num_days - 1) * self.steps_per_day
        params = params_and_v0[:self.model.param_count]
        init_vals = self.model.iv_fn(params_and_v0[self.model.param_count:])
        res = integrate(self.model.rhs_fn, params, init_vals, dt, num)
        return res.T[::self.steps_per_day]

    def obs_x(self, params_and_v0, num_days):
        solution = self.obs_all(params_and_v0, num_days)
        return solution[(-1)]

    def distance(self, params_and_v0):
        cases = self.cases()
        num_days = len(cases)
        solution = self.obs_x(params_and_v0, num_days)
        return np.linalg.norm(solution - cases)

    def cost_fn(self, params_and_v0):
        if not np.less_equal(self.model.bounds[0], params_and_v0).all():
            return 10000000000.0
        else:
            if not np.less_equal(params_and_v0, self.model.bounds[1]).all():
                return 10000000000.0
            return self.distance(params_and_v0) / self.n_sqrt * self.population

    def unbounded_cost_fn(self, params_and_v0):
        return self.distance(params_and_v0) / self.n_sqrt * self.population

    def diff_fn(self, params_and_v0):
        cases = self.cases()
        num_days = len(cases)
        solution = self.obs_x(params_and_v0, num_days)
        return solution - cases

    def random_search(self, search_attempts=1000, update_progress=nop_update_progress, num_best=10, method='lst', optimizer_options={}):
        if method == 'lst':
            return self.random_lst(search_attempts, update_progress, num_best, optimizer_options=optimizer_options)
        if method == 'optimize':
            return self.random_optimize(search_attempts, update_progress, num_best, optimizer_options=optimizer_options)
        raise ValueError('Unsupported search method: ' + method)

    def random_optimize(self, search_attempts=1000, update_progress=nop_update_progress, num_best=10, optimizer_options={}):
        results = []
        best = 1e+100
        for i in range(search_attempts):
            if 1 == i % 10:
                update_progress(i / search_attempts, str(best))
            x = np.random.uniform(self.model.bounds[0], self.model.bounds[1])
            opt = (optimize.minimize)(self.cost_fn, x, method='Nelder-Mead', **optimizer_options)
            if not opt.success:
                pass
            else:
                if opt.fun < best:
                    best = opt.fun
                results.append((opt.fun, opt.x))

        update_progress(1, str(best))
        results = sorted(results, key=(lambda v: v[0]))
        self.random_search_results = np.array(results).T
        self.best = self.random_search_results[:, :num_best]
        self._best_solutions = {}
        return self.random_search_results

    def random_lst(self, search_attempts=1000, update_progress=nop_update_progress, num_best=10, optimizer_options={}):
        results = []
        best = 1e+100
        for i in range(search_attempts):
            if 1 == i % 10:
                update_progress(i / search_attempts, str(best))
            x = np.random.uniform(self.model.bounds[0], self.model.bounds[1])
            lst_res = (optimize.least_squares)(self.diff_fn, x, bounds=self.model.bounds, method='trf', **optimizer_options)
            if lst_res.success:
                cost = self.cost_fn(lst_res.x)
                if cost < best:
                    best = cost
                results.append((cost, lst_res.x))

        update_progress(1, str(best))
        results = sorted(results, key=(lambda v: v[0]))
        self.random_search_results = np.array(results).T
        self.best = self.random_search_results[:, :num_best]
        self._best_solutions = {}
        return self.random_search_results

    def best_solutions(self, num_best=10, future_days=80):
        key = (
         num_best, future_days)
        if key in self._best_solutions:
            return self._best_solutions[key]
        else:
            res = []
            for i in range(num_best):
                best_v = self.best[1][i]
                solution = self.obs_all(best_v, len(self.cases()) + future_days)
                res.append(pd.DataFrame((solution.T), columns=(self.model.solution_spec)))

            self._best_solutions[key] = res
            return res

    def best_table(self):
        table = pd.DataFrame((list(self.best[1])), columns=(self.model.report_columns))
        table['R0'] = table.alpha / table.beta
        table['Precision'] = self.best[0]
        for c in self.model.population_columns:
            table[c] = table[c] * self.population

        styler = table.style.format('{:.3g}')
        for c in self.model.population_columns:
            styler = styler.format({c: '{:.1f}'})

        return styler

    def t_space(self, days_from_now, now_is_zero=True):
        len_cases = len(self.cases())
        if now_is_zero:
            return np.linspace(-len_cases + 1, days_from_now, len_cases + days_from_now)
        else:
            return np.arange(len_cases + days_from_now)

    def days_range(self, after_today=0):
        return np.arange(TODAY - self.ncases(), TODAY + after_today)

    def plot_match(self, best_key, figsize=(10, 10)):
        best_results, days = best_key
        best_solutions = (self.best_solutions)(*best_key)
        cases_t = self.t_space(0, False)
        solution_t = self.t_space(3, False)
        plt.figure(figsize=figsize)
        plt.plot(cases_t, (self.cases() * 100), 'o', label='Real cases')
        for i, solution in enumerate(best_solutions):
            plt.plot(solution_t, (solution['O'][:len(self.cases()) + 3] * 100), '-', label=f"Solution {i + 1}")

        plt.legend()
        plt.title(self.country + ': Real cases vs model, percentage of population')
        plt.gca().set_ylabel('% of population')
        plt.gca().set_xlabel('Days')

    def plot_difference(self, num=5, figsize=(10, 10)):
        days = self.days_range()
        plt.figure(figsize=figsize)
        for i, bs in enumerate(self.best_solutions(num, 0)):
            plt.plot(days, ((bs['O'] - self.cases()) * self.population), label=f"Solution {i + 1}")

        plt.grid()
        plt.legend()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(DATES_FORMATTER)
        plt.setp((ax.get_xticklabels()), rotation=90, ha='right')
        plt.title(self.country + ': Real cases vs model, difference')
        plt.gca().set_ylabel('Population')
        plt.gca().set_xlabel('Dates')

    def plot_prognosis(self, best_key, figsize=(10, 10)):
        best_results, days = best_key
        best_solutions = (self.best_solutions)(*best_key)
        solution_t = self.t_space(days)
        cases_t = self.t_space(0)
        plt.figure(figsize=figsize)
        plt.plot(cases_t, (self.cases() * 100), 'o', label='Real cases')
        for i, solution in enumerate(best_solutions):
            plt.plot(solution_t, (solution['O'] * 100), '-', label=f"{i + 1}: {max(solution['O']) * self.population:.0f}")

        plt.legend()
        plt.title(self.country + ': Model of revealed cases, percentage of population')
        plt.gca().set_ylabel('% of population')
        plt.gca().set_xlabel('Days from today')
        plt.figure(figsize=figsize)
        for i, solution in enumerate(best_solutions):
            plt.plot(solution_t, (solution['I'] * 100), '-', label=f"{i + 1}: {max(solution['I']) * self.population:.0f}")

        plt.legend()
        plt.title(self.country + ': Model of infected, percentage of population')
        plt.gca().set_ylabel('% of population')
        plt.gca().set_xlabel('Days from today')
        plt.figure(figsize=figsize)
        for i, solution in enumerate(best_solutions):
            plt.plot(solution_t, (solution['S'] * 100), '-', label=f"{i + 1}: {min(solution['S']) * self.population:.0f}")

        plt.legend()
        plt.title(self.country + ': Model of susceptible, percentage of population')
        plt.gca().set_ylabel('% of population')
        plt.gca().set_xlabel('Days from today')

    def present_best(self, days=80, best_results=5, figsize=(10, 10)):
        best_key = (
         best_results, days)
        self.plot_match(best_key, figsize=figsize)
        self.plot_difference(best_results, figsize=figsize)
        self.plot_prognosis(best_key, figsize=figsize)

    def run(self, search_attempts=1000, days=80, best_results=5, figsize=(10, 6), method='lst', optimizer_options={}):
        self.random_search(search_attempts, update_progress=update_progress, method=method, optimizer_options=optimizer_options)
        self.present_best(days, best_results, figsize)
        return self.best_table()
