import re
from scipy.spatial import distance

def clean_years(issued_tags):
    years = []
    for issued in issued_tags:
        result = re.findall("\d{4}", issued)
        if result:
            years.extend(result)
    valid_years = set([year for year in years if year.startswith(("19", "20"))])
    print(valid_years)
    

def clean_title(title_tags: list):
    titles = []
    for title in title_tags:
        print(title)
        title_case = title.title()
        print(title_case)
        titles.append(title_case)

    print(titles)
    
    string1 = titles[0]
    string2 = titles[1]
    Normalized_HD = distance.hamming(list(string1),list(string2))
    print("The Normalized Hamming Distance between {} and {} is {}".format(string1, string2,Normalized_HD))
    # Original Hamming Distance
    print("The Hamming Distance between {} and {} is {}".format (string1,string2, Normalized_HD*len(string1)))
    # https://www.researchgate.net/publication/396617636_Lightweight_String_Similarity_Approaches_for_Duplicate_Detection_in_Academic_Titles

def clean_authors(author_tags):
    authors = []
    for author in author_tags:
        split_authors = author.split(", ")
        authors.extend(split_authors)
    
    # Also needs a string similarity thing right here.
    
    unique_authors = set(authors)
    print(unique_authors)
    return unique_authors
        
clean_authors(['M van Noordenne', 'M . van Noordenne', 'M . van Noordenne', 'M . van Noordenne'])


# ('Frank de Beer',)
# ('jan jaap harts', 'Jan Jaap Harts')
# ('F. W. J. Scholten',)
# ('M.J', 'Boskamp', 'M.J .', 'Boskamp')
# ('Henk Baten', 'Henk Baten')
# ('Martin Slabbèrtje', 'Martin Slabbèrtje')
# ('Jeroen de Roos', 'Jeroen de Roos')
# ('Robert Hassink Vianen',)
# ('Stefan de Jong', 'Stefan de de Jong')
# ('N . F . Abcouwer', 'N . F . Abcouwer')
# ('Frank Studulski', 'Hidde Toet')
# ('Eefke Langendonk', 'Eefke Langendonk')
# ('Jan Kennis', 'Martin Schluter', 'Jan Kennis', 'Martin Schluter', 'Jan Kennis', 'Martin Schluter', 'Jan Kennis', 'Martin Schluter')
# ('Rolf v', 'Acquoy', 'Wil Anker', 'Dorothé Bloks', 'Frank Gubbels', 'Rolf Hunink', 'Joep Jilesen', 'Oda v.d .', 'Kemp', 'Dick Korenhof', 'Wim v.d .', 'Maas', 'Roel de Mink', 'Jan Mulder', 'Lody Smeets', 'Ben Steur', 'Monica Thung', 'Saskia Vleer')
# ('EMMY HESSELS', 'JEROEN SMIT', 'EMMY HESSELS', 'JEROEN SMIT')
# ('Klaske W. Ypma', 'Klaske W . Ypma')
# ('PAUL VAN BEUNUM', 'RONALD H.KRANENBURG', 'Paul van Beijnum', 'Ronald H', 'Kranenburg')
# ('M van Noordenne', 'M . van Noordenne', 'M . van Noordenne', 'M . van Noordenne')
# ('VAN ELZAKKER', 'Ad van Elzakker')
# ('Wïlly Dwarkasing',)


 

# clean_title(['ONTWIKKELINGEN IN DE INDUSTRIËLE STRUCTUUR VAN ZUID-WEST-NEDERLAND 1945 - 1970', 'een sociaal-geografische verkenning'])


# For each, there will be some kind of metric with sentence similarity or something that decides how close they are. The longer one can be kept. Also get rid of anything that's dirty, that includes non-real things.


# ('ONTWIKKELINGEN IN DE INDUSTRIËLE STRUCTUUR VAN ZUID-WEST-NEDERLAND 1945 - 1970', 'een sociaal-geografische verkenning', 'ONTWIKKELINGEN IN DE INDUSTRIËLE STRUCTUUR VAN ZUID - WEST - NEDERLAND 1945 - 1970', 'een sociaal-geografische verkenning')
# ('migratie : een verbetering ?', 'VERSLAG VAN EEN HUISHOUDENSENQUETE', 'MIGRATIE : EEN VERBETERING ?')
# ('De Markeverdelingen in Zuidoost-Salland',)
# ('De invloed van persoonskenmerken op verschillende vormen van gebruik van een binnenstad', 'De invloed van persoonskenmerken op verschillende vormen van gebruik van een binnenstad .')
# ('HET WOONGEDRAG IN DE WIJK JUTPHAAS — WIJKERSLOOT - GEMEENTE NIEUWEGEIN',)
# ('een onderzoek naar de koopgerichtheid van huishoudens in nieuwegein', 'GROEIKERN OOK KOOPKERN ?', 'een onderzoek naar de koopgerichtheid van huishoudens in Nieuwegein')
# ('REGIONALE VERSCHILLEN IN MEDISCHE CONSUMPTIE een sociaal-geografisehe studie',)

# ('INNOVATIEBEVORDERING IN BADEN-WUERTTEMBERG Op zoek naar een verklaring voor een economisch succes', 'INNOVATIEBEVORDERING IN BADEN-WUERTTEMBERG Op zoek naar een verklaring voor een economisch succes')
# ('RANDSTAD EN TCLECOTWICATIE', 'Randstad en telecom - municatie')
# ('Vergrijzen in de grote stad',)
# ('HET GROENE HART VAN BINNEN NAAR BUITEN', 'Stemmer < . . it het 1 groen “ , een onderz ■ « k naar de visie va ; de ■ yrootste gemeenten in het middengebied van de Randstad op de ruimtelijke ordening ervan ing .', 'HET GROENE HART VAN BINNEN NAAR BUITEN', "Stemmen uit het ' groen ' , een onderzoek naar de visie van de grootste gemeenten in het middengebied van de Randstad op de ruimtelijke ordening ervan", 'HET GROENE HART VAN BINNEN NAAR BUITEN', "Stemmen uit het ' groen ' , een onderzoek naar de visie van de grootste gemeenten in het middengebied van de Randstad op de ruimtelijke ordening ervan", 'Het Groene Hart van binnen naar buiten', "Stemmen uit het ' groen ' , een onderzoek naar de visie van de grootste gemeenten in het middengebied van de Randstad op de ruimtelijke ordening ervan")
# ('Do . wne 16 de Julio na het iïAM-RLKF-rtoject .', 'DE ZONE 16 DE JÜLIO NA HET HAM-BIRF-PROJECT', 'De effecten van buurtverbetering in El Alto , Bolivia .')
# ('WINKELEN OP DE UITHOF !', 'FICTIE OF WERKELIJKHEID ?')
# ('BEDRMVENTERRE1NENMARKT EN HERSTRUCTURERING : PRIORITEIT OF LAPMIDDEL ?', 'DE STADSREGIONALE BEDRIJVENTERREINENMARKT EN HERSTRUCTURERING : PRIORITEIT OF LAPMIDDEL ?')
# ('Hongarije : land van de twee snelheden ?', 'De gevolgen van de transitie naar een markteco - nomie op de regionale economische ontwikkeling', 'Hongarije : land van de twee snelheden ?', 'De gevolgen van de transitie naar een markteco - nomie op de regionale economische ontwikkeling', 'Hungary : Country of two velocitics ?', 'The consequences of the transition to a market-economy for the regional economie development')
# ('Het PPP-project Hof / Krommestraat als voorbeeldfunctie', 'Het PPP-project Hof / Krommestraat als voorbeeldfunctie')
# ('De Regio Stedendriehoek', 'Regionale ontwikkeling en de betekenis van watersystemen voor de relatie tussen stedelijk en landelijk gebied', 'De Regio Stedendriehoek', 'Regionale ontwikkeling en de betekenis van watersystemen voor de relatie tussen stedelijk en landelijk gebied')
# ('Angstgevoelens tijdens het uitgaan', 'Angstgevoelens tijdens het uitgaan')
# ('E-commerce in de binnenstad Een onderzoek naar e-commerce succes onder winkeliers in de binnenstad', 'E-commerce in de binnenstad Een onderzoek naar e-commerce succes onder winkeliers in de binnenstad')
