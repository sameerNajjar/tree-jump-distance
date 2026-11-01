// Exact prefix branch-and-bound for weighted permutation domains.
// C ABI; no Python callbacks or probabilistic state hashes.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <new>
#include <vector>
#include <limits>
#include <unordered_map>
#include <array>

struct Balance { int8_t net, peak; };
static const std::array<Balance,65536>& balance_table() {
    static const auto table=[] {
        std::array<Balance,65536> out{};
        for(int s=0;s<256;++s)for(int u=0;u<256;++u) {
            int value=0,peak=0;
            for(int j=0;j<8;++j) {
                value+=((s>>j)&1)-((u>>j)&1);peak=std::max(peak,value);
            }
            out[(s<<8)|u]={int8_t(value),int8_t(peak)};
        }
        return out;
    }();
    return table;
}

#if defined(_WIN32)
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

struct DomainResult {
    int32_t status;  // 0 complete; 1 timeout; 2 candidate/memory cap; 3 allocation/error
    uint64_t nodes, pruned, rows;
    double elapsed;
    const int16_t* permutations;
    const int64_t* distances;
    void* owner;
};
struct Storage {
    std::vector<int16_t> permutations;
    std::vector<int64_t> distances;
};

struct Search {
    int n, q, branches;
    const int32_t* weights;
    const int16_t* refs;
    const int64_t* terms; // (branch, ref_a, ref_b or -1, constant, divisor)
    int term_count;
    const int32_t *term_offsets, *term_pairs, *tour_offsets, *tour_pairs;
    int tour_count;
    int64_t budget, W=0;
    uint64_t cap, byte_cap;
    double seconds;
    std::chrono::steady_clock::time_point begin;
    DomainResult& result;
    Storage& storage;
    std::vector<int> rank, preferred;
    std::vector<int64_t> dp, dist, branch_values;
    std::vector<uint8_t> unused;
    std::vector<int16_t> prefix;
    std::vector<int64_t> prefix_dist;
    uint64_t used_mask=0;
    int64_t used_weight=0;
    bool unit=true;
    bool fast=false;
    int bytes=0;
    std::vector<uint64_t> bit_states, unused_bits, block_masks;
    std::vector<int> block_starts;
    std::unordered_map<uint64_t,std::vector<int64_t>> subset_cache;
    static constexpr int64_t neg = std::numeric_limits<int64_t>::min()/4;

    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    }

    // Exact minimum distance to each reference among ALL extensions of prefix.
    static int popcount(uint64_t x) {
        x-= (x>>1)&0x5555555555555555ULL;
        x=(x&0x3333333333333333ULL)+((x>>2)&0x3333333333333333ULL);
        x=(x+(x>>4))&0x0f0f0f0f0f0f0f0fULL;
        return int((x*0x0101010101010101ULL)>>56);
    }

    std::vector<int64_t> pair_costs(uint64_t mask) {
        std::vector<int64_t> out(q*q,0);
        int64_t weight=0;
        for(int s=0;s<n;++s)if(mask&(1ULL<<s))weight+=weights[s];
        for(int a=0;a<q;++a)for(int b=a+1;b<q;++b) {
            int64_t best=0;
            if(unit) {
                uint64_t full=(1ULL<<n)-1,V=full;
                for(int j=0;j<n;++j) {
                    int s=refs[b*n+j];
                    if(!(mask&(1ULL<<s)))continue;
                    uint64_t M=1ULL<<rank[a*n+s];
                    V=((V+(V&M))|(V&~M))&full;
                }
                best=n-popcount(V);
            } else {
                int pos[64], count=0;int64_t ending[64];
                for(int j=0;j<n;++j) {
                    int s=refs[b*n+j];
                    if(!(mask&(1ULL<<s)))continue;
                    int p=rank[a*n+s];int64_t prev=0;
                    for(int k=0;k<count;++k)if(pos[k]<p)prev=std::max(prev,ending[k]);
                    pos[count]=p;ending[count]=prev+weights[s];
                    best=std::max(best,ending[count]);++count;
                }
            }
            out[a*q+b]=out[b*q+a]=weight-best;
        }
        return out;
    }

    // Alphabet partition: cost(full) >= cost(used) + cost(unused).
    bool split_admissible(int depth) {
        if(n>20 || !tour_count || depth<2 || depth>=n-1)return true;
        auto found=subset_cache.find(used_mask);
        std::vector<int64_t> temporary;
        const std::vector<int64_t>* values;
        if(found!=subset_cache.end()) values=&found->second;
        else {
            auto used=pair_costs(used_mask);
            auto remaining=pair_costs(((1ULL<<n)-1)^used_mask);
            temporary.assign(term_count+1,0);
            for(int j=0;j<tour_count;++j) {
                int64_t cost=0;
                for(int k=tour_offsets[j];k<tour_offsets[j+1];++k)cost+=remaining[tour_pairs[k]];
                temporary[0]=std::max(temporary[0],(cost+1)/2);
            }
            for(int j=0;j<term_count;++j)
                for(int k=term_offsets[j];k<term_offsets[j+1];++k)temporary[j+1]+=used[term_pairs[k]];
            // Bounded optional memoization. Cache exhaustion never rejects a state.
            if(subset_cache.size()<65536) values=&subset_cache.emplace(used_mask,std::move(temporary)).first->second;
            else values=&temporary;
        }
        std::fill(branch_values.begin(),branch_values.end(),0);
        for(int j=0;j<term_count;++j) {
            const int64_t* t=terms+5*j;
            int64_t value=prefix_dist[t[1]]+(*values)[j+1];
            if(t[2]>=0)value+=prefix_dist[t[2]];
            if(t[4]==2) value=(value+1)>>1;
            branch_values[t[0]]=std::max(branch_values[t[0]],value);
        }
        int64_t lower=(*values)[0];
        for(auto x:branch_values)lower+=x;
        return lower<=budget;
    }

    bool admissible(int depth) {
        if(fast) {
            const auto& table=balance_table();
            for(int a=0;a<q;++a) {
                uint64_t S=bit_states[depth*q+a], U=unused_bits[depth*q+a];
                int balance=0,peak=0;
                for(int j=0;j<bytes;++j) {
                    auto entry=table[((S&255)<<8)|(U&255)];
                    peak=std::max(peak,balance+entry.peak);balance+=entry.net;
                    S>>=8;U>>=8;
                }
                dist[a]=used_weight-peak;
                if(tour_count)prefix_dist[a]=used_weight-popcount(bit_states[depth*q+a]);
            }
        } else {
        for (int a=0; a<q; ++a) {
            int64_t remaining=0, best=0, prefix_best=0;
            for (int j=n-1; j>=0; --j) {
                int symbol=refs[a*n+j];
                if (unused[symbol]) remaining += weights[symbol];
                else {
                    best=std::max(best, dp[a*n+j]+remaining);
                    prefix_best=std::max(prefix_best,dp[a*n+j]);
                }
            }
            best=std::max(best,remaining); // LCS can use no prefix symbols
            dist[a]=W-best;
            prefix_dist[a]=used_weight-prefix_best;
        }
        }
        std::fill(branch_values.begin(),branch_values.end(),0);
        for (int i=0; i<term_count; ++i) {
            const int64_t* t=terms+5*i;
            int64_t value=dist[t[1]]+t[3];
            if (t[2]>=0) value+=dist[t[2]];
            if(t[4]==2) value=(value+1)>>1;
            branch_values[t[0]]=std::max(branch_values[t[0]],value);
        }
        int64_t lower=0;
        for (auto v:branch_values) lower+=v;
        return lower<=budget && split_admissible(depth);
    }

    void visit(int depth) {
        if (result.status) return;
        ++result.nodes;
        if ((result.nodes & 255)==1 && elapsed()>=seconds) {
            result.status=1; return;
        }
        if (!admissible(depth)) { ++result.pruned; return; }
        if (depth==n) {
            uint64_t count=result.rows+1;
            uint64_t per_row=uint64_t(n)*sizeof(int16_t)+uint64_t(q)*sizeof(int64_t);
            if (count>cap || (byte_cap && count>byte_cap/per_row)) {
                result.status=2; return;
            }
            storage.permutations.insert(storage.permutations.end(),prefix.begin(),prefix.end());
            storage.distances.insert(storage.distances.end(),dist.begin(),dist.end());
            result.rows=count;
            return;
        }
        for (int s:preferred) {
            if (!unused[s]) continue;
            if(fast) {
                for(int a=0;a<q;++a) {
                    uint64_t state=bit_states[depth*q+a];
                    int start=block_starts[a*n+s];
                    for(int j=0;j<weights[s];++j) {
                        uint64_t x=state|(1ULL<<(start+j));
                        state=x&~(x-((state<<1)|1));
                    }
                    bit_states[(depth+1)*q+a]=state;
                    unused_bits[(depth+1)*q+a]=unused_bits[depth*q+a]&~block_masks[a*n+s];
                }
            } else {
            for (int a=0; a<q; ++a) {
                int p=rank[a*n+s];
                int64_t best=0;
                for (int j=0; j<p; ++j) best=std::max(best,dp[a*n+j]);
                dp[a*n+p]=best+weights[s];
            }
            }
            unused[s]=0; prefix[depth]=int16_t(s);
            used_weight+=weights[s];
            if(n<=20)used_mask|=1ULL<<s;
            visit(depth+1);
            unused[s]=1;
            used_weight-=weights[s];
            if(n<=20)used_mask^=1ULL<<s;
            if(!fast)for (int a=0; a<q; ++a) dp[a*n+rank[a*n+s]]=neg;
            if (result.status) return;
        }
    }

    void run() {
        rank.resize(n*q); preferred.resize(n);
        dp.assign(n*q,neg); dist.resize(q); prefix_dist.resize(q); branch_values.resize(branches);
        unused.assign(n,1); prefix.resize(n);
        for (int s=0; s<n; ++s) { W+=weights[s]; preferred[s]=s; if(weights[s]!=1)unit=false; }
        for (int a=0; a<q; ++a)
            for (int j=0; j<n; ++j) rank[a*n+refs[a*n+j]]=j;
        fast=W<=64;bytes=int((W+7)/8);
        if(fast) {
            bit_states.assign((n+1)*q,0);unused_bits.resize((n+1)*q);
            block_masks.resize(n*q);block_starts.resize(n*q);
            uint64_t full=W==64?~0ULL:(1ULL<<W)-1;
            for(int a=0;a<q;++a) {
                unused_bits[a]=full;int start=0;
                for(int j=0;j<n;++j) {
                    int s=refs[a*n+j];block_starts[a*n+s]=start;
                    block_masks[a*n+s]=weights[s]==64?~0ULL:((1ULL<<weights[s])-1)<<start;
                    start+=weights[s];
                }
            }
        }
        std::stable_sort(preferred.begin(),preferred.end(),[&](int x,int y){
            int64_t sx=0,sy=0;
            for(int a=0;a<q;++a){sx+=rank[a*n+x];sy+=rank[a*n+y];}
            return sx<sy;
        });
        visit(0);
    }
};

API DomainResult* enumerate_prefix_domain(int n,int q,int branches,
    const int32_t* weights,const int16_t* refs,const int64_t* terms,int term_count,
    int64_t budget,uint64_t cap,uint64_t byte_cap,double seconds,
    const int32_t* term_offsets,const int32_t* term_pairs,
    const int32_t* tour_offsets,const int32_t* tour_pairs,int tour_count) {
    auto* result=new(std::nothrow) DomainResult{};
    if(!result)return nullptr;
    auto* storage=new(std::nothrow) Storage();
    if(!storage){result->status=3;return result;}
    result->owner=storage;
    auto begin=std::chrono::steady_clock::now();
    try {
        Search search{n,q,branches,weights,refs,terms,term_count,term_offsets,term_pairs,tour_offsets,tour_pairs,tour_count,budget,0,cap,byte_cap,
                      seconds,begin,*result,*storage};
        search.run();
    } catch(...) { result->status=3; }
    result->elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    result->permutations=storage->permutations.data();
    result->distances=storage->distances.data();
    return result;
}

API void release_prefix_domain(DomainResult* result) {
    if(result){delete static_cast<Storage*>(result->owner);delete result;}
}
