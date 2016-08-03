##This is a simple model based on the famous Elo rating adapted for basketball 
##games prediction.

import numpy as np
import pandas as pd


#parameters to account for home advantage
NEUTRAL = 0.0
AWAY = -200.0
HOME = 80.0
#game importance for tournament/regular matches
TOURNIMP = 1.0
REGIMP = 0.15
#game importance based on time passed from the game
YEARIMP = 1988 #games are important starting from this year
TAU = 2
#probability evaluation
PROBPAR = 300.0
##basic Elo
K = 15.0
BETA = 230.0
#modification of Elo to account for the point difference
DELTA = 0.4
GAMMA = 8
#random seed
RND = 190


class Elorating():
   """This is a modification of Elo rating model which accounts for point difference."""
   def __init__(self, k=K, beta=BETA, delta=DELTA, gamma=GAMMA, win=1, lose=0):
      ##k-factor and beta are from the basic elo
      self.k = k
      self.beta = beta
	  ###parameters for the modified version to account for point difference
      self.delta = delta
      self.gamma = gamma
	  ###victory and loss scores
      self.win = win
      self.lose = lose

   def exp_score(self, rating, other_rating, point, other_point, loc, score):
      pd = abs(float(other_point) - float(point))
      if pd == 0:
         pd = 1
      diff = float(other_rating) - float(rating) - loc
      f_factor = 2 * self.beta
      return score + (-1) ** score * self.delta ** (1 + pd / self.gamma) - 1. / (1 + 10 ** (diff / f_factor))
      #return score - 1. / (1 + 10 ** (diff / f_factor))
      #return (pd/25) * ( score - 1. / (1 + 10 ** (diff / f_factor)) )

   def newrating(self, rating, other_rating, point, other_point, imp, loc, score):
      adjustment = self.exp_score(rating, other_rating, point, other_point, loc, score)
      new_rating = float(rating) + imp * self.k * adjustment
      return new_rating

   def rate(self, rating1, rating2, point1, point2, imp, loc):
      ##scores for win and lose
      scores = (self.win, self.lose)
      new_rating1 = self.newrating(rating1, rating2, point1, point2, imp, loc, scores[0])
      new_rating2 = self.newrating(rating2, rating1, point2, point1, imp, loc, scores[1])
      return (new_rating1, new_rating2)



class Prob():
   def __init__(self, probpar=PROBPAR, neutral=NEUTRAL, away=AWAY, home=HOME):
      self.probpar = probpar
	  ###game importance parameter based on the location
      self.neutral = neutral
      self.away = away
      self.home = home

   def probability(self, rating1, rating2):
      return 1/(1+10**((rating2 - rating1) / self.probpar))


   def prediction(self, seasontrain, tcr, test, rscr, preds):
      """The model is trained on all the previous regular games 
      and the previous tournaments"""
      proba = np.zeros((preds.shape[0], 1))
      i = 0
      for S in seasontrain:
         team = {}
         points = {}
         train = {}
         if S > seasontrain[0]:
            tcr = pd.concat((tcr, test.loc[test.Season == S-1]), axis=0, ignore_index=True)
         train = pd.concat((tcr, rscr.loc[rscr.Season == S]), axis=0, ignore_index=True)
         elo = Elorating(RND)
         train.loc[train.Wloc == "N", "Wloc"] = self.neutral
         train.loc[train.Wloc == "A", "Wloc"] = self.away
         train.loc[train.Wloc == "H", "Wloc"] = self.home
         for index, row in train.iterrows():
            t1 = row['Wteam']
            t2 = row['Lteam']
            w = row['Wscore']
            l = row['Lscore']
            Imp = row['Imp']
            Loc = row['Wloc']
            if not t1 in team: 
                team[t1] = 1000.0
            points[t1] = w
            if not t2 in team:
                team[t2] = 1000.0
            points[t2] = l

            (team[t1], team[t2]) = elo.rate(team[t1], team[t2], points[t1], points[t2],Imp, Loc)

         for index, row in preds.iterrows():
            p = list(map(int, str.split(str(row['Id']), '_')))
            if p[0] == S:
               proba[i] = self.probability(team[p[1]], team[p[2]])
               i += 1
      return proba


def validate(test, preds):
   num=test.shape[0]
   valid = np.zeros((num, 1))
   i = 0
   for index, row in test.iterrows():
      wteam = row['Wteam']
      lteam = row['Lteam']
      season = row['Season']
      substr1 = str(season)+'_'+str(wteam)+'_'+str(lteam)
      substr0 = str(season)+'_'+str(lteam)+'_'+str(wteam)
      pred1 = preds.loc[preds.Id == substr1, ['Pred']]
      pred0 = preds.loc[preds.Id == substr0, ['Pred']]
      if not pred1.empty:
          pred = preds.loc[preds.Id == substr1, ['Pred']].values
          s = 1
      elif not pred0.empty:
          pred = preds.loc[preds.Id == substr0, ['Pred']].values
          s = 0
      else: print('There was no such game')
      valid[i] = float(s*np.log(pred)+(1-s)*np.log(1-pred))
      i += 1
   logloss = float(sum(-valid)/num)
   print('Logloss: ', logloss)


def time_scale(tcr, minSeas, tau):
   """Returns an importance value based on the time spent from the game"""
   i = 0
   num=tcr.shape[0]
   Imp_time = np.zeros((num,1))
   for index, row in tcr.iterrows():
      S = row['Season']
      Imp_time[i] = row['Imp'] * (1 - np.exp(-(S-minSeas)/tau))
      i += 1
   return Imp_time

if __name__ == "__main__":

   ##loading data
   tcr = pd.read_csv('./input/TourneyCompactResults.csv')
   tcr['Imp'] = TOURNIMP
   rscr = pd.read_csv('./input/RegularSeasonCompactResults.csv')
   rscr['Imp'] = REGIMP
   ##add time factor to game importance
   Imp_time = time_scale(tcr, YEARIMP, TAU)
   tcr.Imp = Imp_time

   ##Tournaments used for training and validation
   seasontrain = [2012, 2013, 2014, 2015]
   test2012 = tcr.loc[tcr.Season == 2012]
   test2013 = tcr.loc[tcr.Season == 2013]
   test2014 = tcr.loc[tcr.Season == 2014]
   test2015 = tcr.loc[tcr.Season == 2015]
   test = pd.concat((test2012, test2013, test2014, test2015), axis=0, ignore_index=True)
   tcr = tcr[(tcr.Season!=2012)&(tcr.Season!=2013)&(tcr.Season!=2014)&(tcr.Season!=2015)]
   tcr = tcr[(tcr.Season>1988)]

   ##training
   preds = pd.read_csv('./input/SampleSubmission.csv')
   model = Prob(RND)
   proba = model.prediction(seasontrain, tcr, test, rscr, preds)
   preds['Pred'] = np.clip(proba, 0.01, 0.99)

   ##writing
   preds.to_csv('pred_elo_mod.csv', index=False)

   ##validation
   print("Year 2012")
   validate(test2012, preds)
   print("Year 2013")
   validate(test2013, preds)
   print("Year 2014")
   validate(test2014, preds)
   print("Year 2015")
   validate(test2015, preds)
