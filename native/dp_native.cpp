#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <vector>
#include <limits>
#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
using Clock=std::chrono::steady_clock;
struct Limit{};
struct Metric {
    int n; const int32_t* w; int64_t W=0; bool fast;
    std::vector<int> rank;
    std::vector<uint64_t> masks;
    std::vector<int64_t> dp;
    Metric(int n_,const int32_t* w_):n(n_),w(w_),rank(n),masks(n),dp(n) {
        for(int s=0;s<n;++s)W+=w[s]; fast=W<=64;
    }
    static int pc(uint64_t x) {
        x-=(x>>1)&0x5555555555555555ULL;
        x=(x&0x3333333333333333ULL)+((x>>2)&0x3333333333333333ULL);
        x=(x+(x>>4))&0x0f0f0f0f0f0f0f0fULL;
        return int((x*0x0101010101010101ULL)>>56);
    }
    void prepare(const int16_t* a) {
        int start=0;
        for(int j=0;j<n;++j) {
            int s=a[j];rank[s]=j;
            if(fast) { masks[s]=w[s]==64?~0ULL:((1ULL<<w[s])-1)<<start;start+=w[s]; }
        }
    }
    int64_t distance(const int16_t* b) {
        if(fast) {
            uint64_t S=0;
            for(int j=0;j<n;++j) {
                int s=b[j];
                // Each block has w[s] equal expanded letters; the maximum
                // common-subsequence weight equals the weighted block LCS.
                for(int k=0;k<w[s];++k) {
                    uint64_t x=S|masks[s];
                    S=x&~(x-((S<<1)|1));
                }
            }
            return W-pc(S);
        }
        int64_t best=0;
        for(int j=0;j<n;++j) {
            int s=b[j];int64_t v=0;
            for(int k=0;k<j;++k)if(rank[b[k]]<rank[s])v=std::max(v,dp[k]);
            dp[j]=v+w[s];best=std::max(best,dp[j]);
        }
        return W-best;
    }
};
struct Node {
    int pivot=-1,left=-1,right=-1,begin=0,end=0;
    int64_t minF=0,leftLo=0,leftHi=0,rightLo=0,rightHi=0;
};
struct Index {
    int n; int64_t count; const int16_t* B; const int64_t* F;
    Metric metric;
    std::vector<int> ids; std::vector<Node> nodes;
    Clock::time_point deadline;
    uint64_t calls=0,visited=0,pruned=0;
    Index(int n_,int64_t m,const int16_t* b,const int64_t* f,const int32_t* w,double seconds)
      :n(n_),count(m),B(b),F(f),metric(n,w),ids(m),deadline(Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds))) {
        std::iota(ids.begin(),ids.end(),0);nodes.reserve(m/8+8);
    }
    void check() {if(Clock::now()>=deadline)throw Limit();}
    int64_t distance(int j) {
        if((++calls&4095)==0)check();
        return metric.distance(B+int64_t(j)*n);
    }
    int build(int lo,int hi) {
        check();int idx=int(nodes.size());nodes.emplace_back();
        Node node;node.begin=lo;node.end=hi;node.minF=F[ids[lo]];
        int pivotPos=lo;
        for(int k=lo;k<hi;++k)if(F[ids[k]]<node.minF){node.minF=F[ids[k]];pivotPos=k;}
        if(hi-lo<=16) {
            std::sort(ids.begin()+lo,ids.begin()+hi,[&](int a,int b){return F[a]<F[b];});
            nodes[idx]=node;return idx;
        }
        std::swap(ids[lo],ids[pivotPos]);node.pivot=ids[lo];
        metric.prepare(B+int64_t(node.pivot)*n);
        std::vector<std::pair<int64_t,int>> items;
        items.reserve(hi-lo-1);
        for(int k=lo+1;k<hi;++k)items.emplace_back(distance(ids[k]),ids[k]);
        int mid=int(items.size()/2);
        std::nth_element(items.begin(),items.begin()+mid,items.end());
        for(int k=0;k<int(items.size());++k)ids[lo+1+k]=items[k].second;
        auto range=[&](int a,int b,int64_t& low,int64_t& high){
            low=items[a].first;high=low;
            for(int k=a+1;k<b;++k){low=std::min(low,items[k].first);high=std::max(high,items[k].first);}
        };
        range(0,mid,node.leftLo,node.leftHi);range(mid,int(items.size()),node.rightLo,node.rightHi);
        node.left=build(lo+1,lo+1+mid);node.right=build(lo+1+mid,hi);
        nodes[idx]=node;return idx;
    }
    int64_t bound(int child,int64_t d,int64_t lo,int64_t hi) {
        return nodes[child].minF+std::max<int64_t>({0,lo-d,d-hi});
    }
    void query(int idx,int64_t& best,int64_t& arg) {
        ++visited;
        const auto& node=nodes[idx];
        if(node.minF>=best){++pruned;return;}
        if(node.pivot<0) {
            for(int k=node.begin;k<node.end;++k) {
                int j=ids[k];if(F[j]>=best)break;
                int64_t value=F[j]+distance(j);
                if(value<best){best=value;arg=j;}
            }
            return;
        }
        int64_t d=distance(node.pivot),value=F[node.pivot]+d;
        if(value<best){best=value;arg=node.pivot;}
        int a=node.left,b=node.right;
        int64_t ba=bound(a,d,node.leftLo,node.leftHi),bb=bound(b,d,node.rightLo,node.rightHi);
        if(ba>bb){std::swap(a,b);std::swap(ba,bb);}
        if(ba<best)query(a,best,arg);else ++pruned;
        if(bb<best)query(b,best,arg);else ++pruned;
    }
};

// Each row returns its exact minimum if <= limit[row], otherwise limit[row]+1.
// The index is exact: triangle inequality bounds the distance to every point
// in a pivot-distance interval; minF bounds every child's subtree cost.
API int min_plus_index(int n,int64_t na,int64_t nb,const int16_t* A,const int16_t* B,
    const int32_t* weights,const int64_t* F,const int64_t* limits,double seconds,
    int64_t* out,int64_t* args,uint64_t* stats) {
    try {
        Index index(n,nb,B,F,weights,seconds);
        if(!nb) {for(int64_t i=0;i<na;++i){out[i]=limits[i]+1;args[i]=-1;}return 0;}
        int root=index.build(0,int(nb));uint64_t build_calls=index.calls;
        for(int64_t i=0;i<na;++i) {
            if((i&255)==0)index.check();
            out[i]=limits[i]+1;args[i]=-1;
            index.metric.prepare(A+i*n);
            index.query(root,out[i],args[i]);
        }
        stats[0]=build_calls;stats[1]=index.calls-build_calls;stats[2]=index.visited;stats[3]=index.pruned;
        return 0;
    } catch(const Limit&) {return 1;} catch(...) {return 2;}
}

API int median_costs(int n,int64_t count,const int16_t* a,const int16_t* b,const int16_t* C,
    const int32_t* w,double seconds,int64_t* out) {
    try {
        auto deadline=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
        int width=n+1;int64_t W=0;
        for(int s=0;s<n;++s)W+=w[s];
        std::vector<int64_t> prev(int64_t(width)*width),cur(prev.size());
        for(int64_t row=0;row<count;++row) {
            const int16_t* c=C+row*n;
            std::fill(prev.begin(),prev.end(),0);
            for(int i=0;i<=n;++i) {
                if(Clock::now()>=deadline)return 1;
                std::fill(cur.begin(),cur.end(),0);
                for(int j=0;j<=n;++j)for(int k=0;k<=n;++k) {
                    int64_t v=i?prev[j*width+k]:0;
                    if(j)v=std::max(v,cur[(j-1)*width+k]);
                    if(k)v=std::max(v,cur[j*width+k-1]);
                    if(i&&j&&a[i-1]==b[j-1])v=std::max(v,prev[(j-1)*width+k]+w[a[i-1]]);
                    if(i&&k&&a[i-1]==c[k-1])v=std::max(v,prev[j*width+k-1]+w[a[i-1]]);
                    if(j&&k&&b[j-1]==c[k-1])v=std::max(v,cur[(j-1)*width+k-1]+w[b[j-1]]);
                    if(i&&j&&k&&a[i-1]==b[j-1]&&a[i-1]==c[k-1])v=std::max(v,prev[(j-1)*width+k-1]+2*int64_t(w[a[i-1]]));
                    cur[j*width+k]=v;
                }
                prev.swap(cur);
            }
            out[row]=2*W-prev.back();
        }
        return 0;
    } catch(...) {return 2;}
}
