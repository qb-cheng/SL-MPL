# -*- coding: utf-8 -*-


import sys

# User-defined datasets to simulate WF trajectories, including 6 parameters

def GetCaseInfo(SetID):   
    if SetID == 0:
        N = 500
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 1:
        N = 500
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 2:
        N = 500
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 3:
        N = 500
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 10:
        N = 1000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 11:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 12:
        N = 1000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 13:
        N = 1000
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 20:
        N = 2000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 21:
        N = 2000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 22:
        N = 2000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 30:
        N = 5000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 31:
        N = 5000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 32:
        N = 5000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 33:
        N = 5000
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 40:
        N = 10000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 41:
        N = 10000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 42:
        N = 10000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
        
        
    elif SetID == 51:
        N = 20000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 61:
        N = 50000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 71:
        N = 100000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 81:
        N = 100
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
        
        
        
    
        
    
        
        
        
        
        
        
        
    elif SetID == 100:
        N = 500
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 101:
        N = 500
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 102:
        N = 500
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 103:
        N = 500
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 110:
        N = 1000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 111:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 112:
        N = 1000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 113:
        N = 1000
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 120:
        N = 2000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 121:
        N = 2000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 122:
        N = 2000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 130:
        N = 5000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 131:
        N = 5000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 132:
        N = 5000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 133:
        N = 5000
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 140:
        N = 10000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 141:
        N = 10000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
    elif SetID == 142:
        N = 10000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.3
        
        
        
        
    elif SetID == 200:
        N = 500
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 201:
        N = 500
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 202:
        N = 500
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 203:
        N = 500
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 210:
        N = 1000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 211:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 212:
        N = 1000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 213:
        N = 1000
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 220:
        N = 2000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 221:
        N = 2000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 222:
        N = 2000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 230:
        N = 5000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 231:
        N = 5000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 232:
        N = 5000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 233:
        N = 5000
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 240:
        N = 10000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 241:
        N = 10000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
    elif SetID == 242:
        N = 10000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 100000
        InitialAlleleFreq = 0.5
        
        
        
        
        
    elif SetID == 300:
        N = 100
        SC = -0.05
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 301:
        N = 500
        SC = -0.05
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 302:
        N = 1000
        SC = -0.05
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
        
    elif SetID == 310:
        N = 100
        SC = -0.058
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 311:
        N = 500
        SC = -0.058
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 312:
        N = 1000
        SC = -0.058
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
    
    elif SetID == 313:
        N = 2000
        SC = -0.058
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
    
    elif SetID == 314:
        N = 5000
        SC = -0.058
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 315:
        N = 10000
        SC = -0.058
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 316:
        N = 20000
        SC = -0.058
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
    elif SetID == 317:
        N = 50000
        SC = -0.058
        u = 0
        T = 61
        NumItr = 100000
        InitialAlleleFreq = 0.1
        
        
        

    elif SetID == 400:
        N = 10000
        SC = 5/N
        u = 1e-3
        T = 451
        NumItr = 1000
        InitialAlleleFreq = 0.1
        
    elif SetID == 401:
        N = 10000
        SC = 10/N
        u = 1e-3
        T = 451
        NumItr = 1000
        InitialAlleleFreq = 0.1
        
        
    elif SetID == 410:
        N = 100000
        SC = 5/N
        u = 1e-3
        T = 451
        NumItr = 1000
        InitialAlleleFreq = 0.1
        
    elif SetID == 411:
        N = 100000
        SC = 10/N
        u = 1e-3
        T = 451
        NumItr = 1000
        InitialAlleleFreq = 0.1
        
        
        
        
        
    # Reproduce set 11 with 10 times bigger dataset
    elif SetID == 501:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 200000
        InitialAlleleFreq = 0.1
        
    elif SetID == 502:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 200000
        InitialAlleleFreq = 0.1
        
    elif SetID == 503:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 200000
        InitialAlleleFreq = 0.1
        
    elif SetID == 504:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 200000
        InitialAlleleFreq = 0.1
        
    elif SetID == 505:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 200000
        InitialAlleleFreq = 0.1
    
    
    
    
    
    
    elif SetID == 601:
        N = 1000
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 602:
        N = 1000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 603:
        N = 1000
        SC = 0.015
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 604:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 605:
        N = 1000
        SC = 0.025
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 606:
        N = 1000
        SC = 0.03
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 607:
        N = 1000
        SC = 0.035
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 608:
        N = 1000
        SC = 0.04
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 609:
        N = 1000
        SC = 0.045
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 610:
        N = 1000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
        
    elif SetID == 611:
        N = 5000
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 612:
        N = 5000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 613:
        N = 5000
        SC = 0.015
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 614:
        N = 5000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 615:
        N = 5000
        SC = 0.025
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 616:
        N = 5000
        SC = 0.03
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 617:
        N = 5000
        SC = 0.035
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 618:
        N = 5000
        SC = 0.04
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 619:
        N = 5000
        SC = 0.045
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 620:
        N = 5000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
        
    elif SetID == 621:
        N = 10000
        SC = 0.005
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 622:
        N = 10000
        SC = 0.01
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 623:
        N = 10000
        SC = 0.015
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 624:
        N = 10000
        SC = 0.02
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 625:
        N = 10000
        SC = 0.025
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 626:
        N = 10000
        SC = 0.03
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 627:
        N = 10000
        SC = 0.035
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 628:
        N = 10000
        SC = 0.04
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 629:
        N = 10000
        SC = 0.045
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 630:
        N = 10000
        SC = 0.05
        u = 1e-3
        T = 451
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
        
        
    elif SetID == 701:
        N = 1000
        SC = 0.005
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 702:
        N = 1000
        SC = 0.01
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 703:
        N = 1000
        SC = 0.015
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 704:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 705:
        N = 1000
        SC = 0.025
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 706:
        N = 1000
        SC = 0.03
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 707:
        N = 1000
        SC = 0.035
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 708:
        N = 1000
        SC = 0.04
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 709:
        N = 1000
        SC = 0.045
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 710:
        N = 1000
        SC = 0.05
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
        
    elif SetID == 711:
        N = 5000
        SC = 0.005
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 712:
        N = 5000
        SC = 0.01
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 713:
        N = 5000
        SC = 0.015
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 714:
        N = 5000
        SC = 0.02
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 715:
        N = 5000
        SC = 0.025
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 716:
        N = 5000
        SC = 0.03
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 717:
        N = 5000
        SC = 0.035
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 718:
        N = 5000
        SC = 0.04
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 719:
        N = 5000
        SC = 0.045
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 720:
        N = 5000
        SC = 0.05
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
        
    elif SetID == 721:
        N = 10000
        SC = 0.005
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 722:
        N = 10000
        SC = 0.01
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 723:
        N = 10000
        SC = 0.015
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 724:
        N = 10000
        SC = 0.02
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 725:
        N = 10000
        SC = 0.025
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 726:
        N = 10000
        SC = 0.03
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 727:
        N = 10000
        SC = 0.035
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 728:
        N = 10000
        SC = 0.04
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 729:
        N = 10000
        SC = 0.045
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 730:
        N = 10000
        SC = 0.05
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.3
    
        
    
    elif SetID == 801:
        N = 1000
        SC = 0.005
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 802:
        N = 1000
        SC = 0.01
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 803:
        N = 1000
        SC = 0.015
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 804:
        N = 1000
        SC = 0.02
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 805:
        N = 1000
        SC = 0.025
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 806:
        N = 1000
        SC = 0.03
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 807:
        N = 1000
        SC = 0.035
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 808:
        N = 1000
        SC = 0.04
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 809:
        N = 1000
        SC = 0.045
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 810:
        N = 1000
        SC = 0.05
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
        
    elif SetID == 811:
        N = 5000
        SC = 0.005
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 812:
        N = 5000
        SC = 0.01
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 813:
        N = 5000
        SC = 0.015
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 814:
        N = 5000
        SC = 0.02
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 815:
        N = 5000
        SC = 0.025
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 816:
        N = 5000
        SC = 0.03
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 817:
        N = 5000
        SC = 0.035
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 818:
        N = 5000
        SC = 0.04
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 819:
        N = 5000
        SC = 0.045
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 820:
        N = 5000
        SC = 0.05
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
        
    elif SetID == 821:
        N = 10000
        SC = 0.005
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 822:
        N = 10000
        SC = 0.01
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 823:
        N = 10000
        SC = 0.015
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 824:
        N = 10000
        SC = 0.02
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 825:
        N = 10000
        SC = 0.025
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 826:
        N = 10000
        SC = 0.03
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 827:
        N = 10000
        SC = 0.035
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 828:
        N = 10000
        SC = 0.04
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 829:
        N = 10000
        SC = 0.045
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 830:
        N = 10000
        SC = 0.05
        u = 1e-3
        T = 151
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    elif SetID == 1010:
        N = 1000
        SC = 0.01
        u = 1e-3
        T = 901
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 1030:
        N = 5000
        SC = 0.01
        u = 1e-3
        T = 901
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 1040:
        N = 10000
        SC = 0.01
        u = 1e-3
        T = 901
        NumItr = 10000
        InitialAlleleFreq = 0.1
        
    elif SetID == 1110:
        N = 1000
        SC = 0.01
        u = 1e-3
        T = 901
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 1130:
        N = 5000
        SC = 0.01
        u = 1e-3
        T = 901
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 1140:
        N = 10000
        SC = 0.01
        u = 1e-3
        T = 901
        NumItr = 10000
        InitialAlleleFreq = 0.3
        
    elif SetID == 1210:
        N = 1000
        SC = 0.01
        u = 1e-3
        T = 901
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 1230:
        N = 5000
        SC = 0.01
        u = 1e-3
        T = 901
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
    elif SetID == 1240:
        N = 10000
        SC = 0.01
        u = 1e-3
        T = 901
        NumItr = 10000
        InitialAlleleFreq = 0.5
        
        
    else:
        print("Case undefined.")
        sys.exit(1)
        
    return N,u,SC,InitialAlleleFreq,NumItr,T