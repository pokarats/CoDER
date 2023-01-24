#!/bin/sh

# Start from MRCONSO to extract concept list
# This command filters out inactive, nonpreferred concepts and only keeps relevant columns
# awk -F sets FS (field separator) to specified char e.g. awk -F '|' sets '|' as FS
# OFS in the {} part sets FS to what follows the = sign; e.g. {OFS="\t"; print$1,$15} sets \t as the field separator
# for the print function
awk -F '|' '$5=="PF" && $12=="SNOMEDCT_US" && $13=="PT" && $17=="N" {OFS="\t"; print$1,$15}' data/umls/MRCONSO.RRF > data/umls/snomed_active_concepts.txt

# include SCTI as well as UMLS CUI in the active_concepts file (comment out if not needed)
awk -F '|' '$5=="PF" && $12=="SNOMEDCT_US" && $13=="PT" && $17=="N" {OFS="\t"; print$1,$14,$15}' data/umls/MRCONSO.RRF > data/umls/snomed_withscti_active_concepts.txt

# This command filters out inactive relations and only keeps relevant columns
# NB. we reverse $1 and $5 as in UMLS we have (t, r, h), hence make it (h, r, t)
awk -F '|' '$11=="SNOMEDCT_US" && $15=="N" {OFS="\t"; print$5,$4,$1,$8}' data/umls/MRREL.RRF > data/umls/snomed_active_relations.txt

# This command keeps only relevant columsn from MRSTY
awk -F '|' '{OFS="\t"; print$1,$2,$4}' data/umls/MRSTY.RRF > data/umls/snomed_semantic_types.txt

