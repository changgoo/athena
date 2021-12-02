import numpy as np
import pandas as pd
import os
class timing_reader(object):
    def __init__(self,base,pid):
        """ Timing reader class

        Parameters
        ----------
        base : string, base directory name

        pid : string, problem id

        Methods
        -------
        load_task_time
        load_loop_time
        load_timing
        """
        self.base=base
        self.pid=pid
        self.fdict=dict()
        lt = '{}{}.loop_time.txt'.format(base,pid)
        tt = '{}{}.task_time.txt'.format(base,pid)
        if os.path.isfile(lt): self.fdict['loop_time']=lt
        if os.path.isfile(tt): self.fdict['task_time']=tt


    def load_task_time(self,groups=None):
        """Read .task_time.txt file

        Parameters
        ----------
        groups : list, e.g., ['Hydro','Primitives','UserWork']
                 if provided, group tasks that have the same string in the list
                 everything else will be summed and stored in 'Others'

        Returns
        -------
        pandas.DataFrame

        The breakdown of time taken by each task of the time integrator
        """
        def from_block(block):
            info = dict()
            h = block[0].split(',')
            info['ncycle'] = int(h[0].split('=')[1])
            name = h[1].split('=')[1]
            time = h[2].split('=')[1]
            info[name] = float(time)
            for l in block[1:]:
                sp = l.split(',')
                name = sp[0].replace(' ','')
                time = sp[1].split('=')[1]
                info[name] = float(time)
            return info

        with open(self.fdict['task_time']) as fp:
            lines = fp.readlines()

        block_idx = []
        for i,l in enumerate(lines):
            if l.startswith('#'):
                block_idx.append(i)
        timing=dict()

        # initialize
        info = from_block(lines[0:block_idx[1]])

        if groups is None:
            for k in info: timing[k] = []
        else:
            meta = set(['TimeIntegrator','ncycle'])
            keys = set(info.keys()) - meta
            members = dict()
            for g in groups:
                members[g] = []
                for i,k in enumerate(info):
                    if g in k:
                        members[g].append(k)
                        keys = keys - set([k])
            members['Others'] = keys
            for k in  list(meta) + list(members.keys()):
                timing[k] = []

        for i,j in zip(block_idx[:-1],block_idx[1:]):
            info = from_block(lines[i:j])
            if groups is None:
                for k,v in info.items():
                    timing[k].append(v)
            else:
                for g in members:
                    gtime = 0
                    for k in members[g]:
                        gtime += info[k]
                    timing[g].append(gtime)
                for k in meta:
                    timing[k].append(info[k])

        for k in timing:
            timing[k] = np.array(timing[k])
        return pd.DataFrame(timing)


    def load_loop_time(self):
        """Read .loop_time.txt file

        Parameters
        ----------

        Returns
        -------
        pandas.DataFrame

        The breakdown of each step of the main loop including
        Before, TimeIntegratorTaskList, SelfGravity, After
        """
        def from_one_line(line):
            info = dict()
            for sp in line.split(','):
                name, value = sp.split('=')
                if name in ['ncycle']:
                    info[name.replace(' ','')] = int(value)
                else:
                    info[name.replace(' ','')] = float(value)
            return info
        with open(self.fdict['loop_time']) as fp:
            lines = fp.readlines()

        timing = dict()
        info = from_one_line(lines[0])
        for k in info:
            timing[k] = []

        for l in lines:
            info = from_one_line(l)
            for k,v in info.items():
                timing[k].append(v)
        return pd.DataFrame(timing).rename(columns=dict(time='All'))


    def load_timing(self):
        """Read both timing outputs

        Parameters
        ----------

        Returns
        -------
        time in second for each step : xarray.Dataset

        """
        timing1 = self.load_task_time(groups=['Hydro','Field','EMF','Particle','UserWork','Primitives'])
        timing2 = self.load_loop_time()
        timing = timing1.merge(timing2)[['All','Before','SelfGravity','Hydro','Field','EMF',
            'Particle','UserWork','Primitives','Others','After']].to_xarray()
        timing = timing.rename(dict(index='ncycle'))
        return timing
