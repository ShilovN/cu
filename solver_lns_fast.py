import argparse, csv, math, random, time

B=10000; C=300; R=50; D=60; L=10; BAD=-10**15

def load_edges(path):
    g={}
    for line in open(path):
        if line.strip():
            u,v=map(int,line.split())
            g.setdefault(u,set()).add(v); g.setdefault(v,set()).add(u)
    ids=sorted(g); mp={x:i for i,x in enumerate(ids)}
    adj=[[mp[y] for y in g[x]] for x in ids]
    deg=[len(a) for a in adj]
    thr=[math.ceil(.18*d) for d in deg]
    cost=[C*d for d in deg]
    return ids,mp,adj,deg,thr,cost

class Solver:
    def __init__(self,edges,seed=1):
        self.ids,self.mp,self.adj,self.deg,self.thr,self.cost=load_edges(edges)
        self.n=len(self.ids); self.rng=random.Random(seed)

    def empty(self): return [[] for _ in range(D)]
    def cp(self,p): return [x[:] for x in p]

    def read_sub(self,path):
        p=self.empty()
        for r in csv.DictReader(open(path)):
            d=int(r['day']); s=r['node_ids'].strip()
            if s and s!='-1':
                p[d]=[self.mp[int(x)] for x in s.split() if int(x) in self.mp]
        return p

    def write_sub(self,p,out,sample=None):
        rows=[{'day':str(i),'node_ids':'-1'} for i in range(D)]
        if sample:
            try:
                z=list(csv.DictReader(open(sample)))
                if len(z)==D and 'day' in z[0] and 'node_ids' in z[0]: rows=z
            except Exception: pass
        for d in range(D):
            rows[d]['day']=str(d)
            rows[d]['node_ids']=' '.join(str(self.ids[v]) for v in p[d]) if p[d] else '-1'
        with open(out,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['day','node_ids']); w.writeheader(); w.writerows(rows)

    def morning(self,act,cnt):
        new=[i for i in range(self.n) if not act[i] and cnt[i]>=self.thr[i]]
        for v in new: act[v]=1
        for v in new:
            for u in self.adj[v]: cnt[u]+=1
        return new

    def sim(self,p,trace=False):
        act=[0]*self.n; bought=[0]*self.n; cnt=[0]*self.n
        budget=B; spent=0; viral=0; states=[]
        for d in range(D):
            new=self.morning(act,cnt); viral+=len(new); budget+=R*len(new)
            if trace: states.append((act[:],cnt[:],bought[:],budget))
            if len(p[d])>L: return BAD,viral,spent,budget,states
            seen=set()
            for v in p[d]:
                if v in seen or v<0 or v>=self.n or bought[v] or act[v] or budget<self.cost[v]:
                    return BAD,viral,spent,budget,states
                seen.add(v); budget-=self.cost[v]; spent+=self.cost[v]; bought[v]=1; act[v]=1
                for u in self.adj[v]: cnt[u]+=1
        return viral*R-spent,viral,spent,budget,states

    def future(self,act0,cnt0,day):
        act=act0[:]; cnt=cnt0[:]; when={}
        for d in range(day+1,D):
            new=[i for i in range(self.n) if not act[i] and cnt[i]>=self.thr[i]]
            if not new: break
            for v in new: act[v]=1; when[v]=d
            for v in new:
                for u in self.adj[v]: cnt[u]+=1
        return when

    def apply(self,act,cnt,bun):
        a=act[:]; c=cnt[:]
        for v in bun:
            if not a[v]:
                a[v]=1
                for u in self.adj[v]: c[u]+=1
        return a,c

    def quick(self,v,act,cnt,budget):
        if act[v] or self.cost[v]>budget: return -1e9
        s=0.0
        for u in self.adj[v]:
            if act[u]: continue
            need=self.thr[u]-cnt[u]
            if need<=1: s+=900
            elif need==2: s+=220
            elif need==3: s+=60
            for w in self.adj[u][:20]:
                if not act[w]:
                    n2=self.thr[w]-cnt[w]
                    if n2<=1: s+=15
                    elif n2==2: s+=4
        return s/max(1,self.cost[v]/300)+0.5*self.deg[v]

    def pool(self,act,cnt,bought,budget,k,rand=0):
        arr=[v for v in range(self.n) if not act[v] and not bought[v] and self.cost[v]<=budget]
        arr.sort(key=lambda v:self.quick(v,act,cnt,budget),reverse=True)
        out=arr[:k]
        if rand and len(arr)>k:
            rest=arr[k:]; self.rng.shuffle(rest); out+=rest[:rand]
        return list(dict.fromkeys(out))

    def eval_bundle(self,bun,act,cnt,bought,budget,day,base=None):
        bun=tuple(dict.fromkeys(bun))
        if not bun or len(bun)>L: return None
        cost=sum(self.cost[v] for v in bun)
        if cost>budget or any(act[v] or bought[v] for v in bun): return None
        if base is None: base=self.future(act,cnt,day)
        a,c=self.apply(act,cnt,bun); fut=self.future(a,c,day)
        extra=acc=0; wr=0.0
        for x,nd in fut.items():
            od=base.get(x)
            if od is None:
                extra+=1; wr+=R*(1+.35*(D-nd)/D)
            elif nd<od:
                acc+=1; wr+=R*.10*(od-nd)/D
        prof=extra*R-cost; wprof=wr-cost
        roi=wprof/max(1,cost); gamma=1.15-.85*day/(D-1)
        score=wprof if wprof<=0 else wprof*((1+max(0,roi))**gamma)
        return score,prof,wprof,extra,acc,cost,bun

    def best_bundle(self,act,cnt,bought,budget,day,slots,k=260,top=110,pair=28,tri=12,minp=-450,rand=False):
        pl=self.pool(act,cnt,bought,budget,k,20 if rand else 0)
        if not pl: return None
        base=self.future(act,cnt,day); singles=[]
        for v in pl[:top]:
            e=self.eval_bundle((v,),act,cnt,bought,budget,day,base)
            if e: singles.append(e)
        singles.sort(reverse=True,key=lambda e:e[0])
        cand=[e for e in singles if e[1]>=minp or e[2]>0]
        nodes=[e[-1][0] for e in singles]
        if slots>=2:
            ns=nodes[:pair]
            for i,a in enumerate(ns):
                for b in ns[i+1:]:
                    if self.cost[a]+self.cost[b]<=budget:
                        e=self.eval_bundle((a,b),act,cnt,bought,budget,day,base)
                        if e and (e[1]>=minp or e[2]>0): cand.append(e)
        if slots>=3 and tri:
            ns=nodes[:tri]; triples=[]
            for i,a in enumerate(ns):
                for j,b in enumerate(ns[i+1:],i+1):
                    for c in ns[j+1:]: triples.append((a,b,c))
            if rand: self.rng.shuffle(triples)
            for t in triples[:1800]:
                if sum(self.cost[x] for x in t)<=budget:
                    e=self.eval_bundle(t,act,cnt,bought,budget,day,base)
                    if e and (e[1]>=minp or e[2]>0): cand.append(e)
        if not cand: return None
        cand.sort(reverse=True,key=lambda e:(e[0],e[1],e[3]))
        return self.rng.choice(cand[:min(8,len(cand))]) if rand and len(cand)>1 else cand[0]

    def greedy(self,rand=False,k=260,top=110,pair=28,tri=12,minp=-450,verbose=True):
        p=self.empty(); act=[0]*self.n; bought=[0]*self.n; cnt=[0]*self.n; budget=B
        for d in range(D):
            new=self.morning(act,cnt); budget+=R*len(new)
            while len(p[d])<L:
                e=self.best_bundle(act,cnt,bought,budget,d,L-len(p[d]),k,top,pair,tri,max(0,minp) if d>=42 else minp,rand)
                if not e or e[0]<=0: break
                added=False
                for v in e[-1]:
                    if len(p[d])>=L: break
                    if not act[v] and not bought[v] and self.cost[v]<=budget:
                        budget-=self.cost[v]; bought[v]=1; act[v]=1; p[d].append(v); added=True
                        for u in self.adj[v]: cnt[u]+=1
                if not added: break
            if verbose and p[d]: print('day',d,'buy',[self.ids[x] for x in p[d]],'budget',budget)
        return p

    def rebuild_day(self,p,day,k=220,top=90,pair=22,tri=8,minp=-450):
        pref=self.empty()
        for d in range(day): pref[d]=p[d][:]
        score,_,_,_,states=self.sim(pref,trace=True)
        if score<=BAD//2 or len(states)<=day: return p
        act,cnt,bought,budget=states[day]; new=[]
        for v in p[day]:
            if len(new)>=L: break
            if not act[v] and not bought[v] and self.cost[v]<=budget:
                budget-=self.cost[v]; bought[v]=1; act[v]=1; new.append(v)
                for u in self.adj[v]: cnt[u]+=1
        while len(new)<L:
            e=self.best_bundle(act,cnt,bought,budget,day,L-len(new),k,top,pair,tri,minp,True)
            if not e or e[0]<=0: break
            ok=False
            for v in e[-1]:
                if len(new)>=L: break
                if not act[v] and not bought[v] and self.cost[v]<=budget:
                    budget-=self.cost[v]; bought[v]=1; act[v]=1; new.append(v); ok=True
                    for u in self.adj[v]: cnt[u]+=1
            if not ok: break
        p[day]=new; return p

    def improve(self,p,iters=1200,seconds=0,k=220,top=90,pair=22,tri=8,minp=-450):
        best=self.cp(p); bs=self.sim(best)[0]; cur=self.cp(best); cs=bs; t0=time.time()
        def pick_day(q):
            ds=[d for d in range(min(D,50)) if q[d]]
            if ds and self.rng.random()<.75: return self.rng.choice(ds)
            x=self.rng.random(); return min(D-1,int(x*x*50))
        for it in range(1,iters+1):
            if seconds and time.time()-t0>seconds: break
            q=self.cp(cur); d=pick_day(q); r=self.rng.random()
            if r<.45:
                for dd in range(max(0,d-1),min(D,d+2)):
                    if q[dd] and self.rng.random()<.7: q[dd].pop(self.rng.randrange(len(q[dd])))
                q=self.rebuild_day(q,d,k,top,pair,tri,minp)
            elif r<.70:
                ds=[x for x in range(D) if q[x]]
                if ds:
                    s=self.rng.choice(ds); v=q[s].pop(self.rng.randrange(len(q[s])))
                    dst=max(0,min(D-1,s+self.rng.choice([-2,-1,1,2])))
                    if len(q[dst])<L: q[dst].append(v)
                    else: q[s].append(v)
            else:
                if q[d]: q[d].pop(self.rng.randrange(len(q[d])))
                q=self.rebuild_day(q,d,k,top,max(8,pair//2),max(0,tri//2),minp)
            ns=self.sim(q)[0]
            if ns<=BAD//2: continue
            delta=ns-cs; temp=1500*max(.02,1-it/max(1,iters))
            if delta>=0 or self.rng.random()<math.exp(delta/max(1,temp)): cur=q; cs=ns
            if ns>bs:
                best=q; bs=ns; print('new_best',it,bs)
            if it%100==0: print('iter',it,'cur',cs,'best',bs)
        return best,bs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--edges',default='marketing_edges.txt')
    ap.add_argument('--sample')
    ap.add_argument('--initial-submission')
    ap.add_argument('--out',default='submission_lns.csv')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--restarts',type=int,default=1)
    ap.add_argument('--lns-iters',type=int,default=1200)
    ap.add_argument('--time-limit',type=float,default=0)
    ap.add_argument('--pool-size',type=int,default=260)
    ap.add_argument('--exact-top',type=int,default=110)
    ap.add_argument('--pair-top',type=int,default=28)
    ap.add_argument('--triple-top',type=int,default=12)
    ap.add_argument('--min-profit',type=int,default=-450)
    a=ap.parse_args(); bestp=None; best=BAD; beststat=None
    for r in range(a.restarts):
        s=Solver(a.edges,a.seed+r)
        if a.initial_submission:
            p=s.read_sub(a.initial_submission); print('restart',r,'initial',s.sim(p)[0])
        else:
            p=s.greedy(r>0,a.pool_size,a.exact_top,a.pair_top,a.triple_top,a.min_profit,a.restarts==1); print('restart',r,'greedy',s.sim(p)[0])
        p,sc=s.improve(p,a.lns_iters,a.time_limit/max(1,a.restarts) if a.time_limit else 0,
                       max(120,a.pool_size-40),max(50,a.exact_top-20),max(12,a.pair_top-6),max(0,a.triple_top-4),a.min_profit)
        stat=s.sim(p); print('restart',r,'improved',stat[:4])
        if stat[0]>best: best=stat[0]; bestp=p; beststat=stat
    s=Solver(a.edges,a.seed); s.write_sub(bestp,a.out,a.sample)
    print('BEST',beststat[:4],'saved',a.out)
    for d,x in enumerate(bestp):
        if x: print(d,' '.join(str(s.ids[v]) for v in x))
if __name__=='__main__': main()
