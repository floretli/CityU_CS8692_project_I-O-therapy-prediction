
trans_table = "./geneid_symbol_alias.table"
headfile = "./GRCh37ERCC_refseq105_head.csv"

removed_gene = ["DUSP27", "UBXN10-AS1", "LINC01279", "C12orf74", 
"DGCR10", "DGCR9", "NKAIN3-IT1", "LOR", "MIR4697HG", "TNRC6C-AS1", "MIG7", "KIAA1107"]

t_file=open(trans_table,'r')
alias2id={}
symbol2id={}
for l in t_file.readlines() : 
	(gene_id,gene_symbol,gene_alias) = (l.split("\t")[0],l.split("\t")[1],l.split("\t")[2].strip("\n"))

	symbol2id[gene_symbol]=gene_id

	if (gene_alias!="-"):
		for alia in  gene_alias.split("|"):
			alias2id[alia]= gene_id 

t_file.close()

h_file=open(headfile,'r')
allgenes = h_file.readline().split(",")[1:]
count_notfound=0
for g in allgenes:
	g_index=allgenes.index(g)
	if (g in symbol2id.keys()):
		allgenes[g_index] = str(symbol2id[g])
	elif(g in alias2id.keys()) :
		if (g not in removed_gene):
			allgenes[g_index] = str(alias2id[g])
		else:
			del allgenes[g_index]

	else:
		print ("gene",g,"id not found!") 
		count_notfound+=1;

o_file=open("./GRCh37ERCC_refseq105_head_idformat_new.csv",'w')

o_file.write(','.join(allgenes))
o_file.close()

print (count_notfound,"genes not found in total!!!") 

