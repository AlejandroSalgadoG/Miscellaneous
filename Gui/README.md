# PyQt4 GUI Application with Design Patterns

A Python desktop application built with PyQt4 that demonstrates various software design patterns including Factory, Command, and Composite patterns. The project also includes a database schema for a billing system.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Design Patterns](#design-patterns)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Database Setup](#database-setup)
- [Code Examples](#code-examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates the implementation of several object-oriented design patterns in Python using PyQt4 for GUI development. The application creates a simple window with buttons that execute commands, showcasing how to structure GUI applications using proven design patterns.

## Features

- **Abstract Base Classes**: Uses Python's ABC module for defining interfaces
- **Factory Pattern**: Creator classes for instantiating GUI components
- **Command Pattern**: Encapsulates button actions as command objects
- **Composite Pattern**: Hierarchical structure of graphic elements
- **Database Integration**: SQL schema for a billing/invoicing system
- **Educational Examples**: Sample code demonstrating inheritance and abstract classes

## Design Patterns

### 1. **Abstract Factory Pattern** (`Creator.py`)

The Creator classes implement the Factory pattern to create GUI components with predefined configurations:

- `WindowCreator`: Creates configured window instances
- `NewButtonCreator`: Creates buttons with specific commands

### 2. **Command Pattern** (`Command.py`)

Commands encapsulate actions that can be executed by GUI elements:

- `Command`: Abstract base class for all commands
- `NewCommand`: Concrete implementation of a command

### 3. **Composite Pattern** (`Graphic.py`)

Graphic elements are organized in a tree structure:

- `Graphic`: Abstract base class for all graphic elements
- `Window`: Container that can hold multiple child elements
- `Button`: Leaf element that executes commands

## Project Structure

```
Gui/
├── Main.py                      # Application entry point
├── Graphic.py                   # GUI component classes (Composite pattern)
├── Command.py                   # Command pattern implementation
├── Creator.py                   # Factory pattern implementation
├── Database.sql                 # SQL database schema
├── Examples/                    # Educational examples
│   ├── AbstractClass/
│   │   └── AbstractClass.py    # Abstract class demonstration
│   ├── Inheritance/
│   │   └── Inheritance.py      # Inheritance example
│   └── Reader/
│       └── ReadData.py         # Excel file reader utility
└── README.md                    # This file
```

## Requirements

- Python 2.7 or Python 3.x
- PyQt4
- xlrd (for Excel reading example)
- MySQL or MariaDB (for database functionality)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Gui
```

### 2. Install Python Dependencies

#### For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install python-qt4
pip install xlrd
```

#### For macOS:

```bash
brew install pyqt@4
pip install xlrd
```

#### For Windows:

Download and install PyQt4 from the official website, then:

```bash
pip install xlrd
```

### 3. Set Up the Database (Optional)

If you want to use the database functionality:

```bash
mysql -u root -p < Database.sql
```

## Usage

### Running the Main Application

```bash
python Main.py
```

This will launch a simple GUI window titled "Database connection" with a "New" button. Clicking the button will print "This is the new command" to the console.

### Running Examples

#### Abstract Class Example

```bash
cd Examples/AbstractClass
python AbstractClass.py
```

Output: `Guitar playing`

#### Inheritance Example

```bash
cd Examples/Inheritance
python Inheritance.py
```

Output:
```
Instrument playing
Guitar playing a song
```

#### Excel Reader Example

```bash
cd Examples/Reader
python ReadData.py path/to/your/file.xlsx
```

This will read and display the 5th row (index 4) of the first sheet in the Excel file.

## Database Setup

The project includes a SQL schema (`Database.sql`) for a billing system with three tables:

### Tables

1. **clients**: Customer information
   - `id_client` (INT, PRIMARY KEY, AUTO_INCREMENT)
   - `name` (VARCHAR(32))
   - `dir` (VARCHAR(32)) - Address
   - `nit` (VARCHAR(32)) - Tax ID
   - `tel` (INT) - Telephone

2. **bills**: Invoice records
   - `id_bill` (INT, PRIMARY KEY, AUTO_INCREMENT)
   - `id_client` (INT, FOREIGN KEY)
   - `total` (INT)

3. **elements**: Line items on bills
   - `id_element` (INT, PRIMARY KEY, AUTO_INCREMENT)
   - `id_bill` (INT, FOREIGN KEY)
   - `descr` (VARCHAR(64)) - Description
   - `quant` (INT) - Quantity
   - `cost` (INT)

### Database Initialization

```bash
mysql -u root -p < Database.sql
```

This will:
1. Drop the existing `size_db` database (if it exists)
2. Create a new `size_db` database
3. Create the three tables with their relationships

## Code Examples

### Creating a Custom Button

```python
from Graphic import Window, Button
from Command import Command

class MyCommand(Command):
    def execute(self):
        print("Button clicked!")

# Create window
window = Window()
window.set_title("My Application")
window.set_size([100, 100, 400, 300])

# Create button
button = Button(window)
button.set_text("Click Me")
button.set_size([150, 100])
button.set_command(MyCommand())

# Add button to window and display
window.add(button)
window.draw()
```

### Creating a Custom Creator

```python
from Creator import Creator
from Graphic import Button
from Command import Command

class ExitCommand(Command):
    def execute(self):
        sys.exit(0)

class ExitButtonCreator(Creator):
    def create(self, window):
        button = Button(window)
        button.set_text("Exit")
        button.set_size([150, 100])
        button.set_command(ExitCommand())
        return button
```

## API Reference

### Graphic.py

#### `Graphic` (Abstract Base Class)

Abstract base class for all graphic elements.

**Methods:**
- `draw()`: Abstract method that must be implemented by subclasses

#### `Window`

Container for GUI elements.

**Methods:**
- `__init__()`: Initialize a new window
- `add(child)`: Add a child element to the window
- `draw()`: Draw all child elements and display the window
- `set_title(title)`: Set the window title
- `set_size(dimensions)`: Set window geometry [x, y, width, height]

#### `Button`

Button GUI element.

**Methods:**
- `__init__(window)`: Initialize a button within a window
- `draw()`: Connect the button to its command
- `set_command(command)`: Set the command to execute on click
- `set_text(text)`: Set the button label
- `set_size(dimensions)`: Set button position [x, y]

### Command.py

#### `Command` (Abstract Base Class)

Abstract base class for all commands.

**Methods:**
- `execute()`: Abstract method that must be implemented by subclasses

#### `NewCommand`

Example command implementation.

**Methods:**
- `execute()`: Prints "This is the new command" to console

### Creator.py

#### `Creator` (Abstract Base Class)

Abstract base class for all creator factories.

**Methods:**
- `create()`: Abstract method that must be implemented by subclasses

#### `WindowCreator`

Factory for creating configured windows.

**Methods:**
- `create()`: Returns a configured Window instance

#### `NewButtonCreator`

Factory for creating "New" buttons.

**Methods:**
- `create(window)`: Returns a configured Button instance attached to the given window

## Architecture

### Application Flow

```
Main.py
   ↓
1. Initialize QApplication
   ↓
2. Create Window (via WindowCreator)
   ↓
3. Create Button (via NewButtonCreator)
   ↓
4. Add Button to Window
   ↓
5. Draw Window (displays all children)
   ↓
6. Enter Qt event loop
```

### Pattern Relationships

```
Creator (Factory)
    ↓ creates
Graphic (Composite)
    ↓ uses
Command (Command Pattern)
```

## Best Practices

1. **Extending Commands**: Create new command classes by inheriting from `Command` and implementing the `execute()` method
2. **Creating Custom Components**: Inherit from `Graphic` to create new GUI components
3. **Factory Pattern**: Use Creator classes to encapsulate complex object creation logic
4. **Separation of Concerns**: Keep GUI logic separate from business logic using the Command pattern

## Troubleshooting

### PyQt4 Import Error

If you encounter `ImportError: No module named PyQt4`:

```bash
# Ubuntu/Debian
sudo apt-get install python-qt4

# macOS
brew install pyqt@4

# Or use PyQt5 (requires code modifications)
pip install PyQt5
```

### Database Connection Issues

Ensure MySQL/MariaDB is running:

```bash
# Check status
sudo systemctl status mysql

# Start service
sudo systemctl start mysql
```
