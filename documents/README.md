# Documents Directory

This directory contains legal documents for processing by the AI Lab IGNFA - Legal RAG System.

**Note**: Document files are not tracked in Git (they are in `.gitignore`). The descriptions below help understand the document corpus even when the actual files are not available in the repository.

---

## 📄 Document Inventory

### Primary Legislation (Acts)

#### 1. **IFA 1927.pdf** - Indian Forest Act, 1927
- **Type**: Primary legislation
- **Subject**: Historical forest legislation governing forest administration, rights, and offences in India
- **Significance**: Foundation act for forest governance, predates independence
- **Key Topics**: Forest administration, reserved/protected forests, forest offences, penalties

#### 2. **WPA 1972.pdf** - Wild Life (Protection) Act, 1972
- **Type**: Primary legislation
- **Subject**: Protection of wild animals, birds, and plants in India
- **Significance**: Comprehensive wildlife protection legislation
- **Key Topics**: Wildlife sanctuaries, national parks, protected species, hunting prohibition, trade restrictions, penalties

#### 3. **FCA 1980.pdf** - Forest Conservation Act, 1980
- **Type**: Primary legislation
- **Subject**: Conservation of forests and prevention of deforestation
- **Significance**: Critical act requiring central government approval for forest land diversion
- **Key Topics**: Forest land diversion, clearance procedures, compensatory afforestation, penalties

#### 4. **BDA 2002.pdf** - Biological Diversity Act, 2002
- **Type**: Primary legislation
- **Subject**: Conservation of biological diversity and sustainable use of biological resources
- **Significance**: Implements India's obligations under the Convention on Biological Diversity
- **Key Topics**: Biodiversity management committees, access and benefit sharing, intellectual property, traditional knowledge

#### 5. **FRA 2006.pdf** - Scheduled Tribes and Other Traditional Forest Dwellers (Recognition of Forest Rights) Act, 2006
- **Type**: Primary legislation
- **Subject**: Recognition of forest rights of forest dwelling communities
- **Significance**: Historic legislation recognizing rights of forest dwellers and tribal communities
- **Key Topics**: Individual and community forest rights, gram sabha powers, resettlement, conservation

### Secondary Materials (Primers and Handbooks)

#### 6. **Primer on FOREST CONSERVATION ACT, 1980.pdf**
- **Type**: Educational/Primer document
- **Subject**: Comprehensive guide to the Forest Conservation Act, 1980
- **Purpose**: Educational resource explaining FCA provisions, procedures, and implementation
- **Key Topics**: Act overview, clearance procedures, case studies, compliance requirements

#### 7. **handbook-wildlife-law-enforcement-india.pdf**
- **Type**: Handbook/Reference guide
- **Subject**: Wildlife law enforcement procedures and practices in India
- **Purpose**: Practical guide for enforcement officers and practitioners
- **Key Topics**: Wildlife offences, enforcement procedures, investigation techniques, prosecution guidelines

### Educational Materials

#### 8. **Module 1 - INTRODUCTION TO LAW & LEGAL SYSTEM.pdf**
- **Type**: Educational module
- **Subject**: Introduction to law and legal system concepts
- **Purpose**: Foundation course material on legal systems
- **Key Topics**: Legal system basics, sources of law, court structure, legal interpretation

#### 9. **Module 2 - Reading Materials - Philosophy, Principles, Environmental Justice and Pollution Control.pdf**
- **Type**: Educational module
- **Subject**: Environmental law philosophy, principles, and pollution control
- **Purpose**: Advanced course material on environmental jurisprudence
- **Key Topics**: Environmental justice, pollution control principles, sustainable development, regulatory framework

### Presentations and Lectures

#### 10. **EPA.pptx** - Environment Protection Act
- **Type**: Presentation/Educational material
- **Subject**: Environment Protection Act, 1986
- **Format**: PowerPoint presentation
- **Key Topics**: EPA provisions, rule-making powers, pollution control, penalties

#### 11. **Air Pollution.pptx**
- **Type**: Presentation/Educational material
- **Subject**: Air pollution laws and regulations
- **Format**: PowerPoint presentation
- **Key Topics**: Air quality standards, regulatory framework, pollution sources, control measures

#### 12. **Dr. Bishwa Kallyan Dash.pptx** (and duplicate: **Dr. Bishwa Kallyan Dash..pptx**)
- **Type**: Lecture presentation
- **Subject**: Environmental law (exact topic varies)
- **Format**: PowerPoint presentation
- **Author**: Dr. Bishwa Kallyan Dash
- **Purpose**: Academic lecture material

#### 13. **Prof. (Dr.) Nandimath Omprakash V.pdf**
- **Type**: Lecture material or publication
- **Subject**: Environmental law or related field
- **Format**: PDF document
- **Author**: Prof. (Dr.) Nandimath Omprakash V.
- **Purpose**: Academic/research material

#### 14. **Sairam Bhat.pptx**
- **Type**: Lecture presentation
- **Subject**: Environmental law or related field
- **Format**: PowerPoint presentation
- **Author**: Sairam Bhat
- **Purpose**: Academic lecture material

---

## 📊 Document Statistics

**Total Documents**: 15 files

**By Type**:
- Primary Legislation: 5 (PDFs)
- Educational/Primers: 2 (PDFs)
- Educational Modules: 2 (PDFs)
- Presentations/Lectures: 6 (PowerPoint files)

**By Format**:
- PDF: 9 files
- PowerPoint (PPTX): 6 files

---

## 🎯 Document Coverage

This corpus covers the following areas of Indian Environmental Law:

1. **Forest Law**: 
   - Indian Forest Act (1927)
   - Forest Conservation Act (1980)
   - Forest Rights Act (2006)

2. **Wildlife Protection**: 
   - Wild Life Protection Act (1972)
   - Wildlife enforcement handbook

3. **Biodiversity**: 
   - Biological Diversity Act (2002)

4. **Pollution Control**: 
   - Environment Protection Act
   - Air pollution regulations

5. **Legal Education**: 
   - Introduction to law and legal systems
   - Environmental law principles and philosophy

---

## 📝 Usage Notes

- **Supported Formats**: PDF, DOCX, PPTX, XLSX, TXT, MD, JPG, PNG, etc.
- **Processing**: All documents in this directory will be processed when running `python -m raganything.cli process-all`
- **Extraction**: Documents are parsed and text content is extracted to `output/[document-name]/[document-name]_extracted.md`
- **Vectorization**: Text chunks are embedded and stored in the vector database for semantic search

---

## 🔍 Query Examples

Based on this corpus, you can ask questions like:

- "What are the key provisions of the Forest Conservation Act, 1980?"
- "What penalties are prescribed for wildlife offences?"
- "What are the requirements for forest land diversion?"
- "How does the Forest Rights Act recognize community rights?"
- "What is the framework for biological diversity conservation?"
- "What are the principles of environmental justice?"

---

## 📚 Adding Documents

To add new documents:

1. Place the file in this `documents/` directory
2. Supported formats will be automatically detected
3. Run `python -m raganything.cli process-all` to process new documents
4. Or use the Web UI to upload and process documents interactively

**Note**: Remember to update this README when adding significant new documents to help users understand the corpus composition.

---

## ⚠️ Important Notes

- **Document Files Not in Git**: The actual document files are excluded from version control (see `.gitignore`)
- **Vector Database**: Processed embeddings are stored in `output/chroma_db/` (tracked via Git LFS)
- **Privacy**: Ensure documents comply with privacy and copyright requirements before processing
- **Legal Accuracy**: This system processes documents as-is; verify legal accuracy independently for critical applications

---

**Last Updated**: See document file modification dates or processing timestamps in the system
