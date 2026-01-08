from rdflib import Graph, Literal, RDF, URIRef, BNode, Namespace
from rdflib.namespace import DCTERMS, FOAF, XSD, RDFS, OWL, RDF, GEO

from annotations_reader import conll_annotations_reader
from find_geography import find_in_dbpedia, clean_name_find_in_dbpedia, find_in_geonames
import os
from pathlib import Path
import re


def add_triple(subject, predicate, object_):
    # g.bind("ex", EX)
    # donna = URIRef("http://example.org/donna")
    # writer = BNode()
    
    thesis = URIRef(f"http://example.org/thesis/{subject}")
    g.add((thesis, predicate, object_))

def safe_iri(entity):
    return re.sub(r'[^a-zA-Z0-9_/:-]', '_', entity)

def encode_thesis_in_kg(conll):
    reader = conll_annotations_reader(conll)
    tags_dict = reader.get_tags_dict()

    if "author" in tags_dict:
        for author in set([author.title() for author in tags_dict["author"]]):
            add_triple(subject=Path(conll).stem, predicate= DCTERMS.creator, object_=Literal(author))
        
    if "issued" in tags_dict:
        for year in set([int(re.findall('\d{4}', issued)[0]) for issued in tags_dict["issued"]]):
            add_triple(subject=Path(conll).stem, predicate= DCTERMS.issued, object_=Literal(year))
    
    if "title" in tags_dict:
        for title in set([title.title() for title in tags_dict["title"]]):
            add_triple(subject=Path(conll).stem, predicate= DCTERMS.title, object_=Literal(title))
    
    if "spatial" in tags_dict:
        clean_spatials = [spatial.replace(".", "").replace(",", "").strip() for spatial in tags_dict["spatial"]]
        
        missing_geos = []
        for spatial in set(clean_spatials):
            
            label, dbp_uri = find_in_dbpedia(spatial)
            if dbp_uri:
                print(f"Found in DBPedia without cleaning name: {spatial}")
                add_triple(subject=Path(conll).stem, predicate=DCTERMS.spatial, object_=URIRef(safe_iri(spatial)))
                add_triple(subject=URIRef(safe_iri(spatial)), predicate=OWL.sameAs, object_=URIRef(dbp_uri))
                add_triple(subject=URIRef(safe_iri(spatial)), predicate=RDF.type, object_=GEO.SpatialObject)
                add_triple(subject=URIRef(dbp_uri), predicate=RDFS.label, object_=Literal(label))
                
                continue
            
            label, dbp_uri = clean_name_find_in_dbpedia(spatial)
            if dbp_uri: 
                print(f"Found in DBPedia after cleaning name: {spatial}")
                add_triple(subject=Path(conll).stem, predicate=DCTERMS.spatial, object_=URIRef(safe_iri(spatial)))
                add_triple(subject=URIRef(safe_iri(spatial)), predicate=RDF.type, object_=GEO.SpatialObject)
                add_triple(subject=URIRef(safe_iri(spatial)), predicate=SCHEMA.containedInPlace, object_=URIRef(dbp_uri))
                add_triple(subject=URIRef(dbp_uri), predicate=RDFS.label, object_=Literal(label))

                continue
            
            label, dbp_uri = find_in_geonames(spatial)
            if dbp_uri:
                print(f"Found in GeoNames: {spatial}")
                add_triple(subject=Path(conll).stem, predicate=DCTERMS.spatial, object_=URIRef(safe_iri(spatial)))
                add_triple(subject=URIRef(safe_iri(spatial)), predicate=RDF.type, object_=GEO.SpatialObject)
                add_triple(subject=URIRef(safe_iri(spatial)), predicate=OWL.sameAs, object_=URIRef(dbp_uri))
                add_triple(subject=URIRef(dbp_uri), predicate=RDFS.label, object_=Literal(label))

                continue
            
            
            missing_geos .append(spatial)
            print(f"Could not find spatial entity for {spatial}")
        return missing_geos

            # print(db_resorce)
            # if spatial_result:
            #     spatial_uri, hypernym_uri = spatial_result
            #     
            #     
            #     add_triple(subject=URIRef(spatial_uri), predicate=URIRef(GOLD.hypernym), object_=URIRef(hypernym_uri))
                
            # else:
            #     print(f"Could not find spatial entity for {spatial}")

if __name__ == "__main__":
    rdf_path = "C:/Users/carme/OneDrive/Documents/Graduate_Programs/Thesis/student_thesis_subset.ttl"
    
    g = Graph()
    ONT = Namespace("http://www.semanticweb.org/carme/ontologies/2025/9/student-thesis-ontology")
    DBPEDIA = Namespace("http://dbpedia.org/resource/")
    SCHEMA = Namespace("http://schema.org/")
    
    g.bind("ont", ONT)
    g.bind("dbpedia", DBPEDIA)
    g.bind("schema", SCHEMA)
    g.parse(rdf_path)

    files_path = "C:/Users/carme/OneDrive/Documents/Graduate_Programs/Thesis/code/annotated_conll_files/cleaned"
    conlls = [os.path.join(files_path, f) for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f)) and f.endswith(".conll")]

    could_not_find_geo = []

    for conll in conlls[15:]:
        missing_geo = encode_thesis_in_kg(conll)
        could_not_find_geo.extend(missing_geo)
    
    g.serialize(destination=rdf_path)
    # print(g.serialize(format="turtle"))
    # print(g.serialize(format="turtle").decode("utf-8"))


# from rdflib import Graph, Literal, RDF, URIRef
# rdflib knows about quite a few popular namespaces, like W3C ontologies, schema.org etc.

# Create a Graph


# # Create an RDF URI node to use as the subject for multiple triples

# # Add another person

# # Iterate over triples in store and print them out.
# print("--- printing raw triples ---")
# for s, p, o in g:
#     print((s, p, o))

# # For each foaf:Person in the store, print out their mbox property's value.
# print("--- printing mboxes ---")
# for person in g.subjects(RDF.type, FOAF.Person):
#     for mbox in g.objects(person, FOAF.mbox):
#         print(mbox)

# # Bind the FOAF namespace to a prefix for more readable output
# g.bind("foaf", FOAF)

# # print all the data in the Notation3 format
# print("--- printing mboxes ---")
# print(g.serialize(format='n3'))





# ########## TABEA CODE



# from sys import platform
# from owlready2 import *
# import numpy as np
# import pandas as pd
# import types
# import rdflib
# from rdflib.namespace import NamespaceManager
# from rdflib import BNode, Namespace, Graph

# ######################################################################################
# ## This script cam be used to populate the ontology with the extracted evidence
# ## as well as the bibliometrics of the studies.
# ######################################################################################

# ##### Define functions
# # graph = default_world.as_rdflib_graph()
# def AddEvidenceInstancesAndRelations(evidence_df, article_df, harmon_BO_classes = False):
#     for count, value in enumerate(evidence_df['DOI']):

#         ## Create an evidence instance individual
#         # Evid_inst = graph.BNode()  # blanknode
#         #     with onto:
#         #         graph.add((Evid_inst, rdflib.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
#         #                rdflib.URIRef("http://www.w3.org/2002/07/owl#Class")))
#         Evid_inst = onto.EvidenceInstance("ev_inst_" + str(count))


#         ## identify the article in ontology or add details if not yet exists
#         doi = value.replace(".csv", "").replace("_", "/")
#         paper_doi = onto.search(iri="*" + doi)
#         if paper_doi:
#             paper_doi = paper_doi[0]
#         else:
#             doi_idx = np.where(article_df['doi'] == doi)[0]
#             print(doi_idx)
#             paper_doi = onto.ReviewStudy(doi)
#             if len(doi_idx) > 0:
#                 if "meta-analy" in str(article_df['Article.Title'].iloc[doi_idx[0]]).lower():
#                     paper_doi.is_a.append(onto.MetaAnalysis)
#                 if "systematic review" in str(article_df['Article.Title'].iloc[doi_idx[0]]).lower() \
#                         or "systematic literature review" in str(article_df['Article.Title'].iloc[doi_idx[0]]).lower():
#                     paper_doi.is_a.append(onto.SystematicReviewStudy)
#                 paper_doi.hasCitation = [str(article_df['citation'].iloc[doi_idx[0]])]
#                 paper_doi.hasStudyYear = [int(article_df['Publication.Year'].iloc[doi_idx[0]])]
#                 paper_doi.hasTitle = [str(article_df['Article.Title'].iloc[doi_idx[0]])]
#                 paper_doi.hasJournal = [str(article_df['Source.Title'].iloc[doi_idx[0]])]
#                 paper_doi.comment = [str(article_df['citation'].iloc[doi_idx[0]])]
#             else:
#                 paper_doi.comment = [str("https://doi.org/" + doi)]
#             for prop in paper_doi.get_properties():
#                 for DataPropertyValue in prop[paper_doi]:
#                     print(".%s == %s" % (prop.python_name, DataPropertyValue))

#         ## identify the evidence info individuals in the ontology
#         ## and add the object property relations
#         if harmon_BO_classes:
#             BO_class_list = onto.search(iri = "*" + str(evidence_df['BehaviorOptionHarmon'].iloc[count].replace(" ", "_").lower()))
#             if len(BO_class_list) < 1:
#                 with onto:
#                     BO_class = types.new_class(str(evidence_df['BehaviorOptionHarmon'].iloc[count].replace(" ", "_").lower()), (onto.BehaviorChoiceOption,))  # make a class
#                     pass
#             else:
#                 BO_class = BO_class_list[0]
#             try:        ### trows an error when instance and class are called equivalently
#                 BO = BO_class(str(evidence_df['BehaviorOption'].iloc[count].replace(" ", "_").lower()))
#             except:     ## so if that is the case we edit the string with an '_instance'
#                 BO = BO_class(str(evidence_df['BehaviorOption'].iloc[count].replace(" ", "_").lower()+"_instance"))
#         else:
#             BO = onto.BehaviorChoiceOption(evidence_df['BehaviorOption'].iloc[count].replace(" ", "_").lower())
#         BD = onto.BehaviorDeterminant(evidence_df['BehaviorDeterminant'].iloc[count].replace(" ", "_").lower())
#         localMO = evidence_df['Moderator'].iloc[count].replace(" ", "_").lower()
#         if localMO != '-100':
#             MO = onto.BehaviorModerator(evidence_df['Moderator'].iloc[count].replace(" ", "_").lower())
#             MO.moderatorIn.append(Evid_inst)
#         localSGs = evidence_df['Studygroup'].iloc[count].replace("[", "").replace("]","").replace("'","").replace("<", "less than").strip().split(", ")
#         if localSGs != ['-100']:
#             for localSG in localSGs:
#                 SG = onto.IndividualProperty(localSG.replace(" ", "_").lower())
#                 SG.groupStudiedIn.append(Evid_inst)

#         if evidence_df['stat_significance'].iloc[count] == "significant":
#             paper_doi.findsSignificance.append(Evid_inst)
#             if evidence_df['stat_direction'].iloc[count] == "positive":
#                 paper_doi.findsPositiveAssociation.append(Evid_inst)
#             elif evidence_df['stat_direction'].iloc[count] == "negative":
#                 paper_doi.findsNegativeAssociation.append(Evid_inst)
#         elif evidence_df['stat_significance'].iloc[count] == "insignificant":
#             paper_doi.findsInsignificance.append(Evid_inst)
#         if evidence_df['stat_consistency'].iloc[count] == "consistent":
#             paper_doi.findsConsistency.append(Evid_inst)
#         elif evidence_df['stat_consistency'].iloc[count] == "inconsistent":
#             paper_doi.findsInconsistency.append(Evid_inst)
#         if evidence_df['stat_correl'].iloc[count] == "correlated":
#             paper_doi.findsCorrelation.append(Evid_inst)
#         BD.studiedDeterminant.append(Evid_inst)
#         paper_doi.studiesDeterminant.append(BD)
#         Evid_inst.fullSentenceString = [str(evidence_df['Fullsentence'].iloc[count])]
#         # Evid_inst.FullSentenceString = [str(evidence_df['Fullsentence'].iloc[count].replace("&", "and").replace("<", "less than").replace(">", "more than").replace("%", "procent").replace("=","equals").replace(":", ""))]
#         Evid_inst.sentenceNumber = [str(evidence_df['Sentence'].iloc[count])]
#         BO.studiedChoiceOption.append(Evid_inst)


#         ## print the evidence instance properties
#         for prop in Evid_inst.get_properties():
#             for DataPropertyValue in prop[Evid_inst]:
#                 print(".%s == %s" % (prop.python_name, DataPropertyValue))


# # Load the ontology
# os.chdir(r"C:\Users\Tabea\Documents\GitHub\TabeaSonnenschein.github.io\ontologies")

# onto = get_ontology(os.path.join(os.getcwd(),"BehaviorChoiceDeterminantsOntology.owl")).load()

# # Set the correct java binary
# if platform == 'win32':
#     JAVA_EXE = '../lib/jdk-17/bin/java.exe'


# ## load data
# ## article details
# os.chdir(r"D:\PhD EXPANSE\Written Paper\02- Behavioural Model paper\modalchoice literature search")
# meta_review_details = pd.read_csv("metareview_details_short.csv")

# ## evidence instances
# os.chdir(r"D:\PhD EXPANSE\Written Paper\02- Behavioural Model paper")
# # evidence_instances_full = pd.read_csv("unique_evidence_instances_clean2_harmonised_BO_manualclean.csv")
# # evidence_instances_full = pd.read_csv("unique_evidence_instances_clean2_harmonised_BO_BD_unique.csv")
# evidence_instances_full = pd.read_csv("unique_evidence_instances_clean2_harmonised_BO_manualclean_BD_unique_SG_unique.csv")
# evidence_instances_full = evidence_instances_full.drop_duplicates()

# evidence_instances = evidence_instances_full[['DOI', 'Sentence', 'Fullsentence', 'BehaviorOption', 'BehaviorDeterminant', 'Studygroup',
#                                               'Moderator', 'stat_significance', 'stat_consistency', 'stat_direction', 'stat_correl']]

# #AddBOMetaClasses(df=evidence_instances_full)

# AddEvidenceInstancesAndRelations(evidence_df = evidence_instances_full,
#                                  article_df= meta_review_details,harmon_BO_classes = True)

# ## Change the ontology annotation to reflect the behavior choice
# BehaviorChoice = "Choice of Mode of Transport"
# BehaviorChoice_short = "TranspModeChoice"
# # print(onto.comment)
# # onto.comment.append("This ontology has been populated with scientific evidence on the "
# #                     "behavior choice of "+BehaviorChoice+". Therefore the evidence on behavior "
# #                     + "choice options, determinants and statistical associations encompassed in this "
# #                     + "ontology is limited to the choice of" + BehaviorChoice + ".")


# #### want to synchronize reasoner to save inferred facts
# #### despite following the instructions of the documentations, it does not work
# with onto:
#     sync_reasoner_pellet(infer_property_values = True,  debug=0)

# os.chdir(r"C:\Users\Tabea\Documents\GitHub\TabeaSonnenschein.github.io\ontologies")
# graph = default_world.as_rdflib_graph()
# graph.serialize(destination="BehaviorChoiceDeterminantsOntology_"+BehaviorChoice_short+".ttl")

# onto.save(file = os.path.join(os.getcwd(),"extendedBehaviouralModelOntology3.owl"), format = "rdfxml")