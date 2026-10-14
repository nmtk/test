//+------------------------------------------------------------------+
//|  WemofEA.mq5                                                      |
//|  WEMOF — "trade only the moves nothing explains" reversal scalp   |
//+------------------------------------------------------------------+
//
//  A mechanization of the method published by わっきゃい (Wakkyai) as WEMOF
//  (Wakkyai's Entry Model Of FX). Only the publicly described rules are
//  reproduced; no proprietary indicator code is used.
//
//  The published rules, on USD/JPY M5:
//    1. Touching Bollinger +/-3 sigma is ENTRY PREPARATION, not an entry.
//    2. The touch is SKIPPED when the move is explainable — a news headline,
//       a break of a prior high/low, or a break of a key level (round number).
//    3. Only an unexplained overshoot is faded, counter-trend, for a small
//       quick target (~1-2 pips) against a wider stop (~3-8 pips).
//
//  Point 2 is the whole method, so each veto is a separate switch here.
//
//  WHAT THIS EA DOES THAT THE PYTHON RESEARCH VERSION CANNOT
//  ---------------------------------------------------------
//  The repo's `wemof` archetype approximates the news veto with a volatility
//  burst, because it has no point-in-time economic calendar. MT5 ships one, so
//  this EA uses the REAL calendar: it blacks out a window around scheduled
//  events for the symbol's own currencies. That is the faithful filter, not a
//  stand-in. The burst test is kept as an optional second line of defence.
//
//  THE NUMBER THAT DECIDES EVERYTHING
//  ----------------------------------
//  A 1-2 pip target against a 3-8 pip stop is a risk/reward of about 1:3 to
//  1:4 AGAINST you. It only breaks even at a very high win rate:
//
//      break-even win rate = SL / (SL + TP)
//      TP 2.0, SL 5.0  ->  71%        TP 1.0, SL 8.0  ->  89%
//
//  Spread comes straight off the target, so it moves that bar further. At a
//  0.3 pip spread on a 2 pip target, ~15% of every win is gone before the
//  trade starts, and the required win rate goes 71% -> 76%. This is why
//  MaxSpreadPips below is a hard refusal and not a suggestion: on an
//  instrument or at an hour where the spread approaches the target, the
//  strategy has negative expectancy by arithmetic, regardless of the signal.
//
//  STATUS — READ BEFORE RUNNING THIS ON REAL MONEY
//  -----------------------------------------------
//  In the research repo this EA belongs to, the close-exit mechanization of
//  this method was measured on 227,047 bars of real USD/JPY M5 built from tick
//  quotes, at realistic FX costs, and produced a NEGATIVE Sharpe (-0.16) with
//  SPA p = 0.996. The pip-exit version (`wemof_tick`), which is what this EA
//  implements, had not yet returned a verdict when this file was written — its
//  bid/ask path was about half backfilled. So the faithful variant is UNTESTED,
//  not validated. Run it on a demo account and in the Strategy Tester with
//  "Every tick based on real ticks" before it sees a cent.
//
//+------------------------------------------------------------------+
#property copyright "WEMOF mechanization — research use"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

//--- Entry: the Bollinger overshoot -------------------------------------------
input group                "Entry — Bollinger overshoot"
input int      BBPeriod          = 20;      // BB period (published: 20)
input double   BBSigma           = 3.0;     // BB deviation (published: 3.0 — NOT 2.0)

//--- Exits: the published pip bracket -----------------------------------------
input group                "Exit — the published pip bracket"
input double   TakeProfitPips    = 1.5;     // Take profit, pips (published: 1-2)
input double   StopLossPips      = 5.0;     // Stop loss, pips (published: 3-8)
input int      MaxHoldBars       = 6;       // Time stop: close after N bars

//--- The four "is this move explainable?" vetoes -------------------------------
input group                "Vetoes — skip anything with an explanation"
input bool     VetoStructure     = true;    // Skip if it breaks the prior N-bar extreme
input int      StructureWindow   = 40;      // N bars for that extreme
input bool     VetoLevel         = true;    // Skip if sitting on a round number
input int      LevelDecades      = 3;       // Round-number grid fineness
input double   LevelTolerancePips= 0.8;     // "On" the level within this many pips
input bool     VetoNews          = true;    // Skip around scheduled economic events
input int      NewsBeforeMin     = 15;      // Blackout starts N min before
input int      NewsAfterMin      = 15;      // Blackout ends N min after
input bool     NewsHighOnly      = false;   // true = high importance only
input bool     VetoBurst         = true;    // Secondary: skip an abnormal impulse bar
input double   BurstATRMult      = 2.0;     // Impulse = range > N x ATR
input bool     VetoTrend         = false;   // Skip moves that merely extend a trend
input int      TrendWindow       = 100;     // SMA period for that trend

//--- Risk and execution --------------------------------------------------------
input group                "Risk and execution"
input double   Lots              = 0.01;    // Fixed lot size
input double   MaxSpreadPips     = 0.6;     // HARD refusal above this (see header)
input int      MaxSlippagePoints = 10;      // Deviation allowed on market orders
input bool     OneTradePerBar    = true;    // At most one entry per bar
input long     MagicNumber       = 20260214;

CTrade        trade;
int           hBands = INVALID_HANDLE;
int           hATR   = INVALID_HANDLE;
int           hMA    = INVALID_HANDLE;
double        pip    = 0.0;                 // price units in one pip
datetime      lastBarTime      = 0;
datetime      lastEntryBarTime = 0;
datetime      newsWindows[][2];             // cached blackout [from, to] pairs
datetime      newsCacheUntil   = 0;

//+------------------------------------------------------------------+
//| One pip in price units.                                           |
//| A 3- or 5-digit quote prices in tenths of a pip, so a pip is ten  |
//| points there and one point everywhere else. Getting this wrong    |
//| scales every distance in the EA by 10.                            |
//+------------------------------------------------------------------+
double PipSize()
  {
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   return ((digits == 3 || digits == 5) ? point * 10.0 : point);
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   pip = PipSize();
   if(pip <= 0.0)
     {
      Print("FATAL: could not resolve pip size for ", _Symbol);
      return(INIT_FAILED);
     }

   hBands = iBands(_Symbol, _Period, BBPeriod, 0, BBSigma, PRICE_CLOSE);
   hATR   = iATR(_Symbol, _Period, BBPeriod);
   hMA    = iMA(_Symbol, _Period, TrendWindow, 0, MODE_SMA, PRICE_CLOSE);
   if(hBands == INVALID_HANDLE || hATR == INVALID_HANDLE || hMA == INVALID_HANDLE)
     {
      Print("FATAL: indicator handle creation failed");
      return(INIT_FAILED);
     }

   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetDeviationInPoints(MaxSlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);

   // A 1-2 pip target is small enough that many brokers will simply refuse it.
   // SYMBOL_TRADE_STOPS_LEVEL is the minimum distance a TP/SL may sit from the
   // market; if the published target is inside it, this method cannot be traded
   // here as written, and the operator needs to know that now rather than after
   // a run of rejected orders.
   long stopsLevelPts = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double stopsLevelPips = stopsLevelPts * SymbolInfoDouble(_Symbol, SYMBOL_POINT) / pip;
   if(stopsLevelPips > 0.0 && TakeProfitPips <= stopsLevelPips)
     {
      PrintFormat("FATAL: broker minimum stop distance is %.2f pips, but TakeProfitPips "
                  "is %.2f. This broker will reject the published target on %s.",
                  stopsLevelPips, TakeProfitPips, _Symbol);
      return(INIT_FAILED);
     }

   // Expectancy sanity check, stated once so it is in the log next to the results.
   double breakEven = StopLossPips / (StopLossPips + TakeProfitPips) * 100.0;
   PrintFormat("WEMOF: BB(%d, %.1f sigma) TP %.1f / SL %.1f pips -> break-even win rate "
               "%.1f%% before spread. pip=%.5f, broker min stop=%.2f pips.",
               BBPeriod, BBSigma, TakeProfitPips, StopLossPips, breakEven, pip,
               stopsLevelPips);
   if(BBSigma < 2.5)
      Print("WARNING: BBSigma < 2.5. The method is built on rare outliers; at 2 sigma "
            "the touch frequency is roughly 15x higher and the premise does not hold.");

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(hBands != INVALID_HANDLE) IndicatorRelease(hBands);
   if(hATR   != INVALID_HANDLE) IndicatorRelease(hATR);
   if(hMA    != INVALID_HANDLE) IndicatorRelease(hMA);
  }

//+------------------------------------------------------------------+
//| Current spread in pips.                                           |
//+------------------------------------------------------------------+
double SpreadPips()
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   return ((ask - bid) / pip);
  }

//+------------------------------------------------------------------+
//| Distance from a price to the nearest round number, in pips.       |
//|                                                                   |
//| The grid is sized relative to the instrument's own magnitude, so   |
//| the same setting means "every 0.1 yen" on USD/JPY at 150 and       |
//| something proportionate on an instrument quoted in thousands.      |
//+------------------------------------------------------------------+
double DistanceToRoundLevelPips(double price)
  {
   if(price <= 0.0) return(1e9);
   double magnitude = MathFloor(MathLog10(price));
   double step = MathPow(10.0, magnitude - (double)LevelDecades);
   if(step <= 0.0) return(1e9);
   double units = price / step;
   double distance = MathAbs(units - MathRound(units)) * step;
   return (distance / pip);
  }

//+------------------------------------------------------------------+
//| Refresh the economic-calendar blackout windows.                   |
//|                                                                   |
//| This is the veto the original actually describes — a human reading |
//| the headline feed — and the one the Python research version had to |
//| approximate. MT5 publishes the calendar, so the EA can ask it      |
//| directly: any scheduled event for either leg of the pair blacks    |
//| out a window around its release time.                             |
//|                                                                   |
//| Note: the calendar is served by the terminal. In the Strategy      |
//| Tester it is available on recent builds, but if it comes back      |
//| empty there, this veto silently passes everything — which is why   |
//| VetoBurst exists as a second line, and why the log says which.     |
//+------------------------------------------------------------------+
void RefreshNewsWindows()
  {
   datetime now = TimeCurrent();
   if(now < newsCacheUntil) return;
   newsCacheUntil = now + 3600;             // rebuild hourly
   ArrayResize(newsWindows, 0);

   string base  = SymbolInfoString(_Symbol, SYMBOL_CURRENCY_BASE);
   string quote = SymbolInfoString(_Symbol, SYMBOL_CURRENCY_PROFIT);
   string currencies[2];
   currencies[0] = base;
   currencies[1] = quote;

   datetime from = now - 86400;
   datetime to   = now + 86400;

   for(int c = 0; c < 2; c++)
     {
      if(currencies[c] == "") continue;
      MqlCalendarValue values[];
      int n = CalendarValueHistory(values, from, to, NULL, currencies[c]);
      for(int i = 0; i < n; i++)
        {
         MqlCalendarEvent event;
         if(!CalendarEventById(values[i].event_id, event)) continue;
         if(NewsHighOnly && event.importance != CALENDAR_IMPORTANCE_HIGH) continue;
         if(event.importance == CALENDAR_IMPORTANCE_NONE) continue;

         int idx = ArrayRange(newsWindows, 0);
         ArrayResize(newsWindows, idx + 1);
         newsWindows[idx][0] = values[i].time - NewsBeforeMin * 60;
         newsWindows[idx][1] = values[i].time + NewsAfterMin  * 60;
        }
     }
  }

//+------------------------------------------------------------------+
bool InNewsBlackout(datetime when)
  {
   RefreshNewsWindows();
   int total = ArrayRange(newsWindows, 0);
   for(int i = 0; i < total; i++)
      if(when >= newsWindows[i][0] && when <= newsWindows[i][1])
         return(true);
   return(false);
  }

//+------------------------------------------------------------------+
//| Is a `side` overshoot explainable? side = +1 long, -1 short.      |
//| A long candidate is a DOWNSIDE overshoot, so the structure test    |
//| looks at lows for longs and highs for shorts.                     |
//+------------------------------------------------------------------+
bool IsExplained(const MqlRates &rates[], int shift, int side)
  {
   // --- news: the real calendar ---------------------------------------------
   if(VetoNews && InNewsBlackout(rates[shift].time))
      return(true);

   // --- structure: a break of the prior N-bar extreme is a new level being
   //     discovered, not an anomaly. Nothing to fade.
   if(VetoStructure)
     {
      double extreme = (side > 0) ? DBL_MAX : -DBL_MAX;
      for(int i = shift + 1; i <= shift + StructureWindow; i++)
        {
         if(i >= ArraySize(rates)) break;
         if(side > 0) extreme = MathMin(extreme, rates[i].low);
         else         extreme = MathMax(extreme, rates[i].high);
        }
      if(side > 0 && rates[shift].low  < extreme) return(true);
      if(side < 0 && rates[shift].high > extreme) return(true);
     }

   // --- level: order flow clusters on round numbers, so an overshoot there is
   //     explained by stop runs and option barriers rather than by noise.
   if(VetoLevel && DistanceToRoundLevelPips(rates[shift].close) < LevelTolerancePips)
      return(true);

   // --- burst: an impulse far larger than recent range. Secondary to the
   //     calendar, and the only news-like filter that works if the calendar is
   //     unavailable in the tester.
   if(VetoBurst)
     {
      double atr[];
      if(CopyBuffer(hATR, 0, shift + 1, 1, atr) == 1 && atr[0] > 0.0)
        {
         double range = rates[shift].high - rates[shift].low;
         if(range > atr[0] * BurstATRMult) return(true);
        }
     }

   // --- trend: a move that merely extends an established trend is explained by
   //     that trend; fading it fights the dominant flow.
   if(VetoTrend)
     {
      double ma[];
      if(CopyBuffer(hMA, 0, shift, 1, ma) == 1)
        {
         int direction = (rates[shift].close > ma[0]) ? 1 : -1;
         if(direction == -side) return(true);
        }
     }

   return(false);
  }

//+------------------------------------------------------------------+
bool HasOpenPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Time stop. The method is a scalp: it does not sit in a losing      |
//| position hoping. TP and SL rest at the broker, so this only has to |
//| handle the "neither was touched" case.                            |
//+------------------------------------------------------------------+
void EnforceTimeStop()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      int barsHeld = iBarShift(_Symbol, _Period, opened, false);
      if(barsHeld >= MaxHoldBars)
        {
         trade.PositionClose(ticket);
         PrintFormat("time stop: closed #%I64u after %d bars", ticket, barsHeld);
        }
     }
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   EnforceTimeStop();

   // Act on closed bars only. A forming bar's high and low still move, so a
   // touch seen mid-bar may not be there when the bar completes — and the bands
   // themselves shift with every new bar.
   datetime barTime = iTime(_Symbol, _Period, 0);
   if(barTime == lastBarTime) return;
   lastBarTime = barTime;

   if(HasOpenPosition()) return;
   if(OneTradePerBar && lastEntryBarTime == barTime) return;

   // Arithmetic refusal: when the spread approaches the target there is no
   // edge left to win, whatever the chart says.
   double spread = SpreadPips();
   if(spread > MaxSpreadPips) return;
   if(spread >= TakeProfitPips)
     {
      PrintFormat("skip: spread %.2f pips >= target %.2f pips — negative by arithmetic",
                  spread, TakeProfitPips);
      return;
     }

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int need = MathMax(StructureWindow + BBPeriod + 5, TrendWindow + 5);
   if(CopyRates(_Symbol, _Period, 0, need, rates) < need) return;

   double upper[], lower[];
   // shift 1 = the bar that just closed.
   if(CopyBuffer(hBands, 1, 1, 1, upper) != 1) return;
   if(CopyBuffer(hBands, 2, 1, 1, lower) != 1) return;

   // Step 1 — PREPARATION: did the closed bar pierce a band? Wicks count: the
   // method is about the overshoot, not where the bar settled.
   int side = 0;
   if(rates[1].low  < lower[0]) side =  1;    // downside overshoot -> buy
   if(rates[1].high > upper[0]) side = -1;    // upside overshoot   -> sell
   if(side == 0) return;

   // Step 2 — the actual method: skip anything with an explanation.
   if(IsExplained(rates, 1, side)) return;

   // Step 3 — fade it, bracketed.
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   bool   ok = false;

   if(side > 0)
     {
      double tp = NormalizeDouble(ask + TakeProfitPips * pip, digits);
      double sl = NormalizeDouble(ask - StopLossPips   * pip, digits);
      ok = trade.Buy(Lots, _Symbol, 0.0, sl, tp, "wemof");
     }
   else
     {
      double tp = NormalizeDouble(bid - TakeProfitPips * pip, digits);
      double sl = NormalizeDouble(bid + StopLossPips   * pip, digits);
      ok = trade.Sell(Lots, _Symbol, 0.0, sl, tp, "wemof");
     }

   if(ok)
     {
      lastEntryBarTime = barTime;
      PrintFormat("entry %s: unexplained %s overshoot, spread %.2f pips",
                  (side > 0 ? "BUY" : "SELL"),
                  (side > 0 ? "downside" : "upside"), spread);
     }
   else
      PrintFormat("order rejected: retcode=%d %s", trade.ResultRetcode(),
                  trade.ResultRetcodeDescription());
  }
//+------------------------------------------------------------------+
