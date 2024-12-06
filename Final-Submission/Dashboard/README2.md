# TEAM-UNIVERSE: Rwanda Youth Labor Market Analysis

## Project Overview
A comprehensive analysis platform examining Rwanda's youth labor market trends and statistics, featuring interactive visualizations and a chat-based exploration interface.

TEAM-UNIVERSE/
├── pycache/
├── .chainlit/
├── .files/
├── assets/
│ ├── Logo_Rwanda.png
│ ├── rwanda-bg.jpeg
│ ├── script2.js
│ └── styles.css
├── data/
│ ├── README.md
│ └── youth_labour_df_updated.csv
├── utils/
│ ├── pycache/
│ ├── init.py
│ ├── data_processing.py
│ └── graph.py
├── .env
├── .gitignore
├── app.py
├── chainlit.md
├── chat_hypo_nopilot.py
├── config.py
├── requirements.txt
└── youth_data_up_to_35_years.csv


## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Installation
``bash
# Clone repository
git clone https://github.com/[username]/TEAM-UNIVERSE.git
cd TEAM-UNIVERSE

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
# Create .env file with:
API_KEY=your_api_key
DATABASE_URL=your_database_url

# Run application
python app.py
# TEAM-UNIVERSE: Rwanda Youth Labor Analysis

## Core Components

### Data Processing
- `utils/data_processing.py`: Handles data cleaning, transformation and preprocessing
- `utils/graph.py`: Creates statistical visualizations and plots
- Processed data stored in `data/youth_labour_df_updated.csv`
- Raw data available in `youth_data_up_to_35_years.csv`

### Web Interface
- Custom styling with `assets/styles.css`
- Interactive features via `assets/script2.js` 
- Branded assets in `assets/`:
  - `Logo_Rwanda.png`
  - `rwanda-bg.jpeg`

### Chat Interface
- Interactive data exploration via `chat_hypo_nopilot.py`
- Chainlit configuration in `.chainlit/`
- Documentation in `chainlit.md`

## Common Issues & Troubleshooting

### Environment Setup
- **Issue**: Python version mismatch
  - Solution: Ensure Python 3.8+ is installed
  - Verify with `python --version`

- **Issue**: Missing dependencies
  - Solution: Run `pip install -r requirements.txt`
  - Check virtual environment activation

- **Issue**: Environment variables
  - Solution: Verify `.env` file configuration
  - Check all required variables are set

### Data Processing
- **Issue**: File not found errors
  - Solution: Confirm data file paths
  - Check file permissions

- **Issue**: Processing errors
  - Solution: Validate data formats
  - Review processing parameters

### Application Errors
- **Issue**: Application crash
  - Solution: Check error logs
  - Verify configurations
  - Validate environment setup

## Development Guidelines

### Code Standards
- Follow PEP 8 style guide
- Document all functions
- Include error handling
- Write unit tests

### Git Workflow
- Create feature branches
- Write descriptive commits
- Update documentation
- Submit pull requests

## Configuration

### Environment Variables
-API_KEY=your_api_key

### Application Settings
Modify `config.py` for:
- Processing parameters
- Visualization settings
- Application configs

## Data Files

### Processed Data
- `data/youth_labour_df_updated.csv`:
  - Clean labor statistics
  - Processed indicators
  - Ready for analysis

### Raw Data
- `youth_data_up_to_35_years.csv`:
  - Original dataset
  - Requires processing
  - Source for analysis

## Acknowledgments

### Organizations
- National Institute of Statistics Rwanda (NISR)


### Contributors
- Development Team
- Data Providers



## Future Development

### Planned Features
- Enhanced visualizations
- Advanced analytics
- Improved UI/UX
- Extended datasets

### Roadmap
1. Q1 2024: Feature enhancements
2. Q2 2024: Performance optimization
3. Q3 2024: New data integration
4. Q4 2024: Platform scaling

