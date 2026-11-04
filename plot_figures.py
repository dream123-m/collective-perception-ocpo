"""Draw experimental Figures 5-9 directly from the supplied CSV/JSON data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plotting(data_dir=None, output_dir=None, figure='all'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    data_dir=Path(data_dir) if data_dir is not None else ROOT/'data'
    out=Path(output_dir) if output_dir is not None else ROOT/'outputs/figures'
    figure=str(figure)
    if figure not in ('all','5','6','7','8','9'):raise ValueError('Unknown figure.')
    out.mkdir(parents=True,exist_ok=True)
    generated=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig);generated.append(name)
    plt.rcParams.update({'font.family':'Arial','font.size':14,'axes.labelsize':16,
        'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':13,
        'grid.color':'lightgray','grid.alpha':.5,'pdf.fonttype':42,
        'axes.linewidth':1.2,'legend.framealpha':1.})
    if figure in ('all','5'):
        curves=pd.read_csv(data_dir/'fig5_training.csv')
        fig,ax=plt.subplots(figsize=(6.4,4.8));x=np.arange(1,1501)
        for method,color in [('mappo','#1f77b4'),('ocpo','#ff7f0e'),('maddpg','#2ca02c')]:
            values=np.stack([curves[(curves.method==method)&(curves.seed==s)].reward.rolling(20,center=True,min_periods=1).mean() for s in (42,123,456)])
            mean,sd=values.mean(0),values.std(0,ddof=1)
            ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.2,linewidth=0)
            ax.plot(x,mean,color=color,linewidth=2,label=method.upper())
        ax.set(xlabel='Episodes',ylabel='Rewards',xlim=(0,1500),ylim=(2,5.5),
               xticks=np.arange(0,1501,300),yticks=np.arange(2,5.51,.5))
        ax.tick_params(which='both',direction='in',top=True,right=True);ax.grid(True,color='lightgray',alpha=.5)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=3,borderaxespad=0,edgecolor='lightgray')
        save(fig,'fig5')
    plt.rcParams.update({'axes.titlesize':16,'axes.linewidth':1.2,'lines.linewidth':2.,
        'ps.fonttype':42,'legend.framealpha':1.,'svg.fonttype':'none'})
    if figure in ('all','6','7'):
        grid=pd.read_csv(data_dir/'fig6_7_summary.csv')
        methods=[('naive','NAIVE','#1F77B4'),('maddpg','MADDPG','#996600'),('mappo','MAPPO','#D62728'),('ocpo','OCPO','#2CA02C')]
        for number,target in ((6,.90),(7,.95)):
            if figure!='all' and figure!=str(number):continue
            for index,pen in enumerate((.3,.5,.7)):
                selected=grid[np.isclose(grid.target,target)&np.isclose(grid.penetration,pen)]
                fig,ax=plt.subplots(figsize=(6.4,4.8));x=np.arange(3)*1.2;width=.18
                for i,(method,label,color) in enumerate(methods):
                    values=[selected[(selected.density==d)&(selected.method==method)].payload_mean.iloc[0] for d in (25,50,75)]
                    ax.bar(x+i*width,values,width,label=label,color=color,edgecolor='black',linewidth=1.2,
                           hatch='///' if method=='ocpo' else None,zorder=3)
                ax.set_xlabel('Nominal Candidate Density (vehs/lane/km)',labelpad=8)
                ax.set_ylabel('Average Payload Size',labelpad=8)
                ax.set_xticks(x+3*width/2,['25','50','75'])
                ax.set_ylim(0,float(selected.payload_mean.max())+6)
                ax.tick_params(direction='in');ax.set_axisbelow(True)
                ax.grid(True,which='major',axis='both',linestyle='-',color='gray',alpha=.3,linewidth=.8)
                ax.legend(loc='upper left',ncol=2,frameon=True,edgecolor='black',fancybox=False,columnspacing=1.,handletextpad=.6)
                save(fig,f'fig{number}{chr(97+index)}')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':8,
        'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'legend.fontsize':7,'axes.linewidth':.8,'lines.linewidth':1.45,'pdf.fonttype':42})
    if figure in ('all','8'):
        summary=read(data_dir/'fig8_summary.json')
        fig,ax=plt.subplots(figsize=(3.5,2.65));right=ax.twinx();limits={'coverage':[],'payload':[]}
        for method,style,alpha in [('mappo',':',.7),('ocpo','-',1.)]:
            rows=sorted((r for r in summary['aggregate'] if r['method']==method),key=lambda r:r['speed_kmh'])
            speeds=[r['speed_kmh'] for r in rows]
            for axis,key,color,marker in [(ax,'coverage','#1f77b4','o'),(right,'payload','#b45f06','s')]:
                means=np.array([r[key+'_mean'] for r in rows]);sd=np.array([r[key+'_sample_sd'] for r in rows])
                axis.errorbar(speeds,means,yerr=sd,linestyle=style,color=color,marker=marker,markersize=3.6,capsize=2.,alpha=alpha,markerfacecolor=color if method=='ocpo' else 'white')
                limits[key].extend((means-sd).tolist()+(means+sd).tolist())
        ax.axhline(.95,color='#777777',linestyle='--',linewidth=.9,alpha=.75)
        ax.set(xlabel='Vehicle speed (km/h)',ylabel='Achieved coverage',xticks=[0,60,90,120])
        right.set_ylabel('Payload size per CAV',color='#b45f06');ax.yaxis.label.set_color('#1f77b4')
        ax.tick_params(axis='y',labelcolor='#1f77b4');right.tick_params(axis='y',labelcolor='#b45f06')
        ax.set_ylim(min(.90,np.floor(min(limits['coverage'])*50)/50),max(1.,np.ceil(max(limits['coverage'])*50)/50))
        right.set_ylim(max(0,np.floor(min(limits['payload'])*2)/2),np.ceil(max(limits['payload'])*2)/2)
        ax.legend(handles=[Line2D([0],[0],color='#333333',linestyle='-',label='OCPO'),Line2D([0],[0],color='#333333',linestyle=':',label='MAPPO')],loc='upper right',ncol=2,frameon=False,handlelength=2.)
        ax.grid(axis='y',linestyle='--',alpha=.3,linewidth=.5)
        for axis in (ax,right):axis.tick_params(direction='in',length=3,width=.8)
        save(fig,'fig8')
    if figure in ('all','9'):
        data=pd.read_csv(data_dir/'fig9_per_cav.csv');summary=read(data_dir/'fig9_summary.json')
        fig,ax=plt.subplots(figsize=(3.5,2.55));colors={.8:'#4C78A8',.85:'#54A24B',.9:'#F58518',.95:'#D62728'}
        rng=np.random.default_rng(2026)
        for target,color in colors.items():
            group=data[np.isclose(data.target,target)]
            ax.scatter(np.full(len(group),target)+rng.normal(0,.0025,len(group)),group.payload,s=10,alpha=.36,color=color,edgecolors='none',clip_on=False)
        rows=sorted(summary['aggregate'],key=lambda r:r['target'])
        line=ax.plot([r['target'] for r in rows],[r['payload_mean'] for r in rows],color='#222222',linewidth=1.5,marker='D',markersize=4.2,markerfacecolor='white',markeredgecolor='#222222',label='Mean payload')[0]
        for row in rows:ax.text(row['target'],row['payload_mean']+.17,f"{row['payload_mean']:.2f}",ha='center',va='bottom',fontsize=7)
        ax.set(xlabel='Target coverage requirement',ylabel='Payload size',xticks=list(colors),xlim=(.775,.975),ylim=(0,13))
        ax.set_xticklabels([f'{x:.2f}' for x in colors])
        handles=[line]+[Line2D([0],[0],linestyle='none',marker='o',markersize=4,markerfacecolor=c,markeredgecolor='none',label=fr'$P_{{\mathrm{{target}}}}={t:.2f}$') for t,c in colors.items()]
        ax.legend(handles=handles,loc='upper left',ncol=2,frameon=True,borderpad=.3,labelspacing=.35,handlelength=1.6,handletextpad=.5,columnspacing=.8)
        ax.tick_params(direction='in',length=3,width=.8);ax.grid(axis='y',linestyle='--',alpha=.3,linewidth=.5)
        save(fig,'fig9')
    print(f'Generated {len(generated)} PDF panels and PNG previews in {out}.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',choices=['all','5','6','7','8','9'],default='all')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/figures')
    args=parser.parse_args()
    plotting(args.data_dir,args.output_dir,args.figure)


if __name__=='__main__':main()
