/* ========================================================================

   (C) Copyright 2023 by Molly Rocket, Inc., All Rights Reserved.
   
   This software is provided 'as-is', without any express or implied
   warranty. In no event will the authors be held liable for any damages
   arising from the use of this software.
   
   Please see https://computerenhance.com for more information
   
   ======================================================================== */

/* ========================================================================
   LISTING 91
   ======================================================================== */

#include "platform_metrics.c"


#ifndef PROFILER
#define PROFILER 0
#endif


#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))

typedef struct {
    char const *Label;
    u64 OldTSCElapsedInclusive;
    u64 StartTSC;
    u32 ParentIndex;
    u32 AnchorIndex;
} profile_block;

#if PROFILER

struct profile_anchor
{
    u64 TSCElapsedExclusive; // NOTE(casey): Does NOT include children
    u64 TSCElapsedInclusive; // NOTE(casey): DOES include children
    u64 HitCount;
    char const *Label;
    u8 IsCounted;
    u64 InnerCount;
};

typedef struct profile_anchor profile_anchor;

static profile_anchor GlobalProfilerAnchors[4096];
static u32 GlobalProfilerParent;

profile_block new_profile_block(profile_block *blk, char const *Label_, u32 AnchorIndex_) {
    //printf("init!\n");
    blk->ParentIndex = GlobalProfilerParent;
        
    blk->AnchorIndex = AnchorIndex_;
    blk->Label = Label_;

    profile_anchor *Anchor = GlobalProfilerAnchors + blk->AnchorIndex;
    blk->OldTSCElapsedInclusive = Anchor->TSCElapsedInclusive;
        
    GlobalProfilerParent = blk->AnchorIndex;
    blk->StartTSC = ReadCPUTimer();
}
    
void end_profile_block(profile_block blk) {
    //printf("exit!\n");
    u64 Elapsed = ReadCPUTimer() - blk.StartTSC;
    GlobalProfilerParent = blk.ParentIndex;

    profile_anchor *Parent = GlobalProfilerAnchors + blk.ParentIndex;
    profile_anchor *Anchor = GlobalProfilerAnchors + blk.AnchorIndex;
    
    Parent->TSCElapsedExclusive -= Elapsed;
    Anchor->TSCElapsedExclusive += Elapsed;
    Anchor->TSCElapsedInclusive = blk.OldTSCElapsedInclusive + Elapsed;
    Anchor->IsCounted = 0;
    ++Anchor->HitCount;
    
    /* NOTE(casey): This write happens every time solely because there is no
        straightforward way in C++ to have the same ease-of-use. In a better programming
        language, it would be simple to have the anchor points gathered and labeled at compile
        time, and this repetitive write would be eliminated. */
    Anchor->Label = blk.Label;
}

void end_counted_profile_block(profile_block blk, int count) {
    u64 Elapsed = ReadCPUTimer() - blk.StartTSC;
    GlobalProfilerParent = blk.ParentIndex;

    profile_anchor *Parent = GlobalProfilerAnchors + blk.ParentIndex;
    profile_anchor *Anchor = GlobalProfilerAnchors + blk.AnchorIndex;
    
    Parent->TSCElapsedExclusive -= Elapsed;
    Anchor->TSCElapsedExclusive += Elapsed;
    Anchor->TSCElapsedInclusive = blk.OldTSCElapsedInclusive + Elapsed;
    Anchor->IsCounted = 1;
    Anchor->InnerCount += count;
    ++Anchor->HitCount;
    
    /* NOTE(casey): This write happens every time solely because there is no
        straightforward way in C++ to have the same ease-of-use. In a better programming
        language, it would be simple to have the anchor points gathered and labeled at compile
        time, and this repetative write would be eliminated. */
    Anchor->Label = blk.Label;
}



#define NameConcat2(A, B) A##B
#define NameConcat(A, B) NameConcat2(A, B)
#define TimeBlock(blk, Name) new_profile_block(&blk, Name, __COUNTER__ + 1);
#define EndTimeBlock(blk) end_profile_block( blk );
#define EndCountedTimeBlock(blk, count) end_counted_profile_block(blk, count);
#define ProfilerEndOfCompilationUnit static_assert(__COUNTER__ < ArrayCount(GlobalProfilerAnchors), "Number of profile points exceeds size of profiler::Anchors array")

static void PrintTimeElapsed(u64 TotalTSCElapsed, profile_anchor *Anchor)
{
    f64 Percent = 100.0 * ((f64)Anchor->TSCElapsedExclusive / (f64)TotalTSCElapsed);
    printf("  %s[%llu]: %llu cycles (%.2f%%", Anchor->Label, Anchor->HitCount, Anchor->TSCElapsedExclusive, Percent);
    if(Anchor->TSCElapsedInclusive != Anchor->TSCElapsedExclusive)
    {
        f64 PercentWithChildren = 100.0 * ((f64)Anchor->TSCElapsedInclusive / (f64)TotalTSCElapsed);
        printf(", %.2f%% w/children", PercentWithChildren);
    }
    if(Anchor->IsCounted) {
        printf(", %llu per iteration", Anchor->TSCElapsedInclusive/Anchor->InnerCount);
    }
    printf(")\n");
}

static void PrintAnchorData(u64 TotalCPUElapsed)
{
    for(u32 AnchorIndex = 0; AnchorIndex < ArrayCount(GlobalProfilerAnchors); ++AnchorIndex)
    {
        profile_anchor *Anchor = GlobalProfilerAnchors + AnchorIndex;
        if(Anchor->TSCElapsedInclusive)
        {
            PrintTimeElapsed(TotalCPUElapsed, Anchor);
        }
    }
}

struct profiler
{
    u64 StartTSC;
    u64 EndTSC;
};
typedef struct profiler profiler;
static profiler GlobalProfiler;

#define TimeFunction TimeBlock(__func__)

static void BeginProfile(void)
{
    GlobalProfiler.StartTSC = ReadCPUTimer();
}

static void EndAndPrintProfile()
{
    GlobalProfiler.EndTSC = ReadCPUTimer();
    u64 CPUFreq = EstimateCPUTimerFreq();
    
    u64 TotalCPUElapsed = GlobalProfiler.EndTSC - GlobalProfiler.StartTSC;
    
    if(CPUFreq)
    {
        printf("\nTotal time: %0.4fms (CPU freq %llu)\n", 1000.0 * (f64)TotalCPUElapsed / (f64)CPUFreq, CPUFreq);
    }
    
    PrintAnchorData(TotalCPUElapsed);
}


static void PrintBlockData(profile_block* blk) { 
    u64 TotalTSCElapsed = GlobalProfiler.EndTSC - GlobalProfiler.StartTSC;
    profile_anchor *Anchor = GlobalProfilerAnchors + blk->AnchorIndex;
    if(Anchor->TSCElapsedInclusive) {
        f64 Percent = 100.0 * ((f64)Anchor->TSCElapsedExclusive / (f64)TotalTSCElapsed);
        printf("  %s[%llu]: %llu (%.2f%%", Anchor->Label, Anchor->HitCount, Anchor->TSCElapsedExclusive, Percent);
        if(Anchor->TSCElapsedInclusive != Anchor->TSCElapsedExclusive)
        {
            f64 PercentWithChildren = 100.0 * ((f64)Anchor->TSCElapsedInclusive / (f64)TotalTSCElapsed);
            printf(", %.2f%% w/children", PercentWithChildren);
        }
        if(Anchor->IsCounted) {
            printf(", %llu per iteration", Anchor->TSCElapsedInclusive/Anchor->InnerCount);
        }
        printf(")\n");
    }
}
#else
#define TimeBlock(...) do { } while(0)
#define PrintAnchorData(...)
#define ProfilerEndOfCompilationUnit
#define EndTimeBlock(blk)
#define EndCountedTimeBlock(blk, count)

void end_profile_block(profile_block blk) {
}
static void EndAndPrintProfile() {
}
#endif
