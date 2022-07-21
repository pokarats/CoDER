/* 
 * File:   main.cpp
 * Author: Aris Kosmopoulos
 *
 * Created on November 9, 2011, 3:24 PM
 */

#include <cstdlib>
#include <iostream>
#include "graph.h"
#include "distanceCopmuter.h"
#include <fstream>
#include <string>
#include <map>
#include <set>
using namespace std;

/*
 * 
 */
void newMeasures(int argc, char** argv,vector<double* >& resultsPerInstance);
void hierPrecRecF1 (int argc, char** argv, vector<double* >& resultsPerInstance);

int main(int argc, char** argv) {            
        
    vector<double* > resultsPerInstance;
    hierPrecRecF1 (argc,argv,resultsPerInstance);
    newMeasures(argc, argv, resultsPerInstance);   
   
    if (argc == 7) {
        ofstream foutResultsPerInstance(argv[6]);
        for (int i=0;i<resultsPerInstance.size();i++) {
            double* la = resultsPerInstance[i];
            foutResultsPerInstance << la[0] << " " << la[1] << " " << la[2] << " " << la[3] << " " << la[4] << " " << la[5] << " " << la[6] << " " << la[7] << endl;            
        }
        foutResultsPerInstance.flush();
        foutResultsPerInstance.close();    
    }
    for (int i=0;i<resultsPerInstance.size();i++) {
        delete [] resultsPerInstance[i];
    }
    return 0;
}

void newMeasures(int argc, char** argv, vector<double* >& resultsPerInstance) {
    Graph G (argv[1]);
    ifstream finGS (argv[2]);
    ifstream finPr (argv[3]);
    
    
    
    int maximumDistance  = stringToInt(argv[4]);
    int maxError = stringToInt(argv[5]);
    string line1,line2;
    double sum = 0.0;
    double sumP = 0.0;
    double sumR = 0.0;
    double MGIAdistance = 0.0;
    int count = 0;
    while (getline(finGS,line1)) {
        if (line1 != "") {
	    
            getline(finPr,line2);
            set<int> gsLCA;
            set<int> prLCA;
	    set<int> gsMGIA = fillSet(line1);
            set<int> prMGIA = fillSet(line2);
	    
	    set<int> Allancs;    
	    set<int>::iterator s_iter,s_iter2;
            for (s_iter=prMGIA.begin();s_iter!=prMGIA.end();s_iter++) {                
                    set<int> ancs = G.getAllAncestors(*s_iter,maximumDistance);
		    
                    Allancs = addSets (Allancs,ancs);                
            }
            for (s_iter=prMGIA.begin();s_iter!=prMGIA.end();s_iter++) {
	      s_iter2 = Allancs.find(*s_iter);
	      if (s_iter2 == Allancs.end())
		prLCA.insert(*s_iter);
	    }
	    Allancs.clear();
	    for (s_iter=gsMGIA.begin();s_iter!=gsMGIA.end();s_iter++) {                
                    set<int> ancs = G.getAllAncestors(*s_iter,maximumDistance);                    
                    Allancs = addSets (Allancs,ancs);                
            }
            for (s_iter=gsMGIA.begin();s_iter!=gsMGIA.end();s_iter++) {
	      s_iter2 = Allancs.find(*s_iter);
	      if (s_iter2 == Allancs.end())
		gsLCA.insert(*s_iter);
	    }
	    double p=0.0;
	    double r=0.0;
	    double LCA_F=0.0;
	    double MGIA=0.0;
	    
	    if (gsLCA.size() != gsMGIA.size() || prLCA.size() != prMGIA.size()) {
	      DistanceComputer dcLCA (G,prLCA,gsLCA, maximumDistance,maxError);
	      DistanceComputer dcMGIA (G,prMGIA, gsMGIA, maximumDistance,maxError);
	      LCA_F = dcLCA.getF(p,r);	      
	      sum += LCA_F;
	      sumP += p;	      
	      sumR += r;
	      MGIA = dcMGIA.getMGIA();
	      MGIAdistance += MGIA;
	    }
	    else {	     
	      DistanceComputer dc (G,prMGIA, gsMGIA, maximumDistance,maxError);
	      
	      LCA_F = dc.getF(p,r);	      
	      sum += LCA_F;
	      sumP += p;	      
	      sumR += r;
	      
	      MGIA = dc.getMGIA();	      
	      MGIAdistance += MGIA;
	      
	    }
            if (argc==7) {
                resultsPerInstance[count][4] = LCA_F;
		resultsPerInstance[count][5] = p;
		resultsPerInstance[count][6] = r;
                resultsPerInstance[count][7] = MGIA;
            }            
            count ++;    
        }
    }    
    cout << "LCA F = "<< (sum/count) << endl;
    cout << "LCA P = "<< (sumP/count) << endl;
    cout << "LCA R = "<< (sumR/count) << endl;
    cout << "MGIA = " << (MGIAdistance/count) << endl;
    finGS.close();
    finPr.close();
}

void hierPrecRecF1 (int argc, char** argv, vector<double* >& resultsPerInstance) {
    Graph G (argv[1]);
    //cout << "Number of G nodes: " << G.getG().size() <<endl;   
    ifstream finGS (argv[2]);
    ifstream finPr (argv[3]);
    int maximumDistance = stringToInt(argv[4]);
    string line1,line2;
    double sumPre = 0.0;
    double sumRec = 0.0;
    double sumF = 0.0;
    //double HammingDist = 0.0;
    int HammingDist = 0;
    int count = 0;
    
    map<int,set<int> > ancestors;
    map<int,set<int> >::iterator anc_iter;
    set<int>::iterator s_iter;
    while (getline(finGS,line1)) {
        if (line1 != "") {
            
            getline(finPr,line2);
            set<int> gs = fillSet(line1);
            set<int> pr = fillSet(line2);
            
            set<int> AncsP;
            set<int> AncsT;            
            
            for (s_iter=gs.begin();s_iter!=gs.end();s_iter++) {
                anc_iter = ancestors.find(*s_iter);
                if (anc_iter == ancestors.end()) {
                    set<int> ancs = G.getAllAncestors(*s_iter,maximumDistance);
                    ancs.insert((*s_iter));
                    ancestors[*s_iter] = ancs;                    
                    AncsT = addSets (AncsT,ancs);
                }
                else
                    AncsT = addSets (AncsT,anc_iter->second);
            }
                
            for (s_iter=pr.begin();s_iter!=pr.end();s_iter++) {
                anc_iter = ancestors.find(*s_iter);
                if (anc_iter == ancestors.end()) {
                    set<int> ancs = G.getAllAncestors(*s_iter,maximumDistance);
                    ancs.insert((*s_iter));
                    ancestors[*s_iter] = ancs;
                    AncsP = addSets (AncsP,ancs);
                }
                else
                    AncsP = addSets (AncsP,anc_iter->second);
            }    
            set<int> PTInter = getIntrOfSets (AncsP,AncsT);
	    /*
	    s_iter = PTInter.find(0);
	    if (s_iter!=PTInter.end())
	      PTInter.erase(s_iter);
	    s_iter = AncsP.find(0);
	    if (s_iter!=AncsP.end())
	      AncsP.erase(s_iter);
	    s_iter = AncsT.find(0);
	    if (s_iter!=AncsT.end())
	      AncsT.erase(s_iter);
	    */
	    
//            cout << "PTInter = " <<  PTInter.size() << endl;
//            cout << "AncsP = " <<  AncsP.size() << endl;
//            cout << "AncsT = " <<  AncsT.size() << endl;
            double precision = (PTInter.size()/((double) AncsP.size()));            
            double recall = (PTInter.size()/((double) AncsT.size()));
            double F1 = 0.0;
            if (precision != 0.0 || recall != 0.0)
                F1 = ((2*precision*recall)/(precision+recall));
            sumF += F1;
            sumPre += precision;
            sumRec += recall;
            
            //ForHammingDistance
            set<int> uniqueP = getSubsetMinusCommon (AncsP,PTInter);
            set<int> uniqueT = getSubsetMinusCommon (AncsT,PTInter);
            double HD = (uniqueP.size() + uniqueT.size());                                                
            HammingDist += HD;
            if (argc==7) {
                double* resultsArray = new double[8];                
                resultsArray[0] = precision;
                resultsArray[1] = recall;
                resultsArray[2] = F1;
                resultsArray[3] = HD;
                resultsPerInstance.push_back(resultsArray);
            }
            count ++;
        }
    }     
    cout << "Hierarchical Precision = " << (sumPre/count) << endl;
    cout << "Hierarchical Recall = " << (sumRec/count) << endl;
    cout << "Hierarchical F1  = " << (sumF/count) << endl;
    cout << "SDL with all ancestors = " << ((double)HammingDist/count) << endl;    
    finGS.close();
    finPr.close();
}
