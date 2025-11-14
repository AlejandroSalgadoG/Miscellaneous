# Design Patterns Implementation

A comprehensive collection of **Gang of Four (GoF) Design Patterns** implemented in multiple programming languages.

## 📋 Table of Contents

- [Overview](#overview)
- [Patterns Implemented](#patterns-implemented)
  - [Creational Patterns](#creational-patterns)
  - [Structural Patterns](#structural-patterns)
- [Project Structure](#project-structure)
- [Languages](#languages)
- [Getting Started](#getting-started)
- [Pattern Descriptions](#pattern-descriptions)
- [UML Diagrams](#uml-diagrams)
- [Building and Running](#building-and-running)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository contains practical implementations of classic design patterns from the seminal book "Design Patterns: Elements of Reusable Object-Oriented Software" by the Gang of Four (Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides).

Each pattern is implemented in **three different programming languages**:
- **C#**
- **C++**
- **Java**

This multi-language approach helps developers understand how the same design pattern can be applied across different programming paradigms and language features.

## 🏗️ Patterns Implemented

### Creational Patterns

Creational patterns deal with object creation mechanisms, trying to create objects in a manner suitable to the situation.

| Pattern | Description | Status |
|---------|-------------|--------|
| [Abstract Factory](#abstract-factory) | Provides an interface for creating families of related objects | ✅ Implemented |
| [Builder](#builder) | Separates object construction from its representation | ✅ Implemented |
| [Factory Method](#factory-method) | Defines an interface for creating objects, letting subclasses decide which class to instantiate | ✅ Implemented |
| [Prototype](#prototype) | Creates new objects by copying an existing object | ✅ Implemented |
| [Singleton](#singleton) | Ensures a class has only one instance | ✅ Implemented |

### Structural Patterns

Structural patterns deal with object composition, creating relationships between objects to form larger structures.

| Pattern | Description | Status |
|---------|-------------|--------|
| [Adapter](#adapter) | Allows incompatible interfaces to work together | ✅ Implemented |
| [Bridge](#bridge) | Decouples an abstraction from its implementation | ✅ Implemented |
| [Composite](#composite) | Composes objects into tree structures to represent hierarchies | ✅ Implemented |
| [Decorator](#decorator) | Adds new functionality to objects dynamically | ✅ Implemented |

## 📁 Project Structure

```
Patterns/
├── C#/
│   ├── Creational/
│   │   ├── AbstractFactory/
│   │   ├── Builder/
│   │   ├── Factory/
│   │   ├── Prototype/
│   │   └── Singleton/
│   └── Structural/
│       ├── Adapter/
│       ├── Bridge/
│       ├── Composite/
│       └── Decorator/
├── C++/
│   ├── Creational/
│   │   ├── AbstractFactory/
│   │   ├── Builder/
│   │   ├── Factory/
│   │   ├── Prototype/
│   │   └── Singleton/
│   └── Structural/
│       ├── Adapter/
│       ├── Bridge/
│       ├── Composite/
│       └── Decorator/
├── Java/
│   ├── Creational/
│   │   ├── AbstractFactory/
│   │   ├── Builder/
│   │   ├── Factory/
│   │   ├── Prototype/
│   │   └── Singleton/
│   └── Structural/
│       ├── Adapter/
│       ├── Bridge/
│       ├── Composite/
│       └── Decorator/
└── Diagrams/
    ├── Raw/          # StarUML .mdj files
    └── Uml/          # Exported UML diagrams (JPG)
```

## 💻 Languages

### C#
- **Compiler**: .NET SDK (Mono or .NET Core)
- **Build Tool**: Makefiles included in each pattern directory

### C++
- **Compiler**: g++ or clang++
- **Standard**: C++11 or higher
- **Build Tool**: Makefiles included in each pattern directory

### Java
- **Compiler**: JDK 8 or higher
- **Build Tool**: Makefiles included in each pattern directory

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed on your system:

#### For C# Examples
```bash
# Install .NET SDK or Mono
# macOS
brew install mono

# Linux (Ubuntu/Debian)
sudo apt install mono-complete

# Windows
# Download from https://dotnet.microsoft.com/
```

#### For C++ Examples
```bash
# macOS
xcode-select --install

# Linux (Ubuntu/Debian)
sudo apt install build-essential

# Windows
# Install MinGW or Visual Studio
```

#### For Java Examples
```bash
# macOS
brew install openjdk

# Linux (Ubuntu/Debian)
sudo apt install default-jdk

# Windows
# Download from https://www.oracle.com/java/technologies/downloads/
```

### Running Examples

Each pattern directory contains a `Makefile` for easy compilation and execution.

```bash
# Navigate to a specific pattern directory
cd Java/Creational/Factory/

# Build and run
make
make run

# Clean up compiled files
make clean
```

## 📖 Pattern Descriptions

### Creational Patterns

#### Abstract Factory

**Intent**: Provide an interface for creating families of related or dependent objects without specifying their concrete classes.

**Example Implementation**: The Abstract Factory pattern creates families of `House` and `Apartment` objects with different styles (Black and White). The factory decides which concrete classes to instantiate.

**Use When**:
- A system should be independent of how its products are created
- A system should be configured with one of multiple families of products
- A family of related product objects is designed to be used together

**Structure**:
```
FactoryCreator → AbstractFactory (FactoryBlack, FactoryWhite)
                 → Products (House, Apartment)
                   → Concrete Products (BlackHouse, WhiteHouse, BlackApartment, WhiteApartment)
```

#### Builder

**Intent**: Separate the construction of a complex object from its representation, allowing the same construction process to create different representations.

**Example Implementation**: Constructs different `Outfit` objects by combining different clothing items (`Coat`, `Hat`). Different builders create different combinations (Jacket+Cap, Vest+Beret, etc.).

**Use When**:
- The algorithm for creating a complex object should be independent of the parts
- The construction process must allow different representations for the object

**Structure**:
```
Director → Builder (interface)
           → ConcreteBuilders (BuilderJacketCap, BuilderVestBeret, etc.)
           → Product (Outfit)
```

#### Factory Method

**Intent**: Define an interface for creating an object, but let subclasses decide which class to instantiate.

**Example Implementation**: Creates different musical `Instrument` objects (Drum, Guitar, Trumpet) through their respective factory classes.

**Use When**:
- A class can't anticipate the class of objects it must create
- A class wants its subclasses to specify the objects it creates
- Classes delegate responsibility to one of several helper subclasses

**Structure**:
```
FactoryCreator → InstrumentFactory (abstract)
                 → ConcreteFactories (DrumFactory, GuitarFactory, TrumpetFactory)
                 → Products (Drum, Guitar, Trumpet)
```

#### Prototype

**Intent**: Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.

**Example Implementation**: Clones `Color` objects (Blue, Red, Yellow) for a `Background`. Instead of creating new color objects, existing ones are cloned.

**Use When**:
- The classes to instantiate are specified at runtime
- Avoiding building a class hierarchy of factories
- Instances of a class can have only a few different combinations of state

**Structure**:
```
Color (Prototype) → ConcretePrototypes (Blue, Red, Yellow)
Background → uses cloned Color objects
```

#### Singleton

**Intent**: Ensure a class has only one instance and provide a global point of access to it.

**Example Implementation**: The `Id` class ensures only one instance exists throughout the application lifecycle.

**Use When**:
- There must be exactly one instance of a class
- The sole instance should be extensible by subclassing
- You need controlled access to a single instance

**Structure**:
```
Singleton (Id) → private constructor
               → static getInstance() method
               → static instance variable
```

### Structural Patterns

#### Adapter

**Intent**: Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.

**Example Implementation**: Adapts different communication methods (`Call`, `Write`) to work through a common `Communication` interface.

**Use When**:
- You want to use an existing class with an incompatible interface
- You want to create a reusable class that cooperates with unrelated classes
- You need to use several existing subclasses but adapting their interface by subclassing is impractical

**Structure**:
```
Communication (Target Interface)
    → Adapter implementations (Call, Write)
      → Adaptee (CellPhone)
```

#### Bridge

**Intent**: Decouple an abstraction from its implementation so that the two can vary independently.

**Example Implementation**: Separates `Animal` types (Cat, Dog) from their `Movement` behaviors (Jump, Run), allowing any animal to have any movement.

**Use When**:
- You want to avoid a permanent binding between abstraction and implementation
- Both abstractions and implementations should be extensible by subclassing
- Changes in the implementation should not impact clients

**Structure**:
```
Animal (Abstraction) → Movement (Implementation)
  ↓                      ↓
Cat, Dog           Jump, Run
```

#### Composite

**Intent**: Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions uniformly.

**Example Implementation**: Represents an organizational hierarchy where `Boss` can contain `Assistant` objects, and `Assistant` can contain `Worker` objects, all implementing the `Employee` interface.

**Use When**:
- You want to represent part-whole hierarchies of objects
- You want clients to be able to ignore the difference between compositions and individual objects

**Structure**:
```
Employee (Component)
    → Composite (Boss, Assistant)
    → Leaf (Worker)
```

#### Decorator

**Intent**: Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.

**Example Implementation**: Adds powers to superheroes dynamically. A `SuperHero` can be decorated with various powers like `Fly` and `XRay` vision through the `HeroPower` decorator.

**Use When**:
- You want to add responsibilities to individual objects dynamically
- Extension by subclassing is impractical
- You want to be able to withdraw responsibilities

**Structure**:
```
SuperHero (Component)
    → HiperMan (ConcreteComponent)
    → HeroPower (Decorator)
      → Fly, XRay (ConcreteDecorators)
```

## 🎨 UML Diagrams

Each pattern includes UML diagrams to help visualize the structure and relationships:

- **Raw Diagrams**: StarUML `.mdj` files located in `Diagrams/Raw/`
- **Exported Diagrams**: JPG images located in `Diagrams/Uml/`

### Viewing Diagrams

```bash
# Creational Patterns
open Diagrams/Uml/Creational/AbstractFactory.jpg
open Diagrams/Uml/Creational/Builder.jpg
open Diagrams/Uml/Creational/Factory.jpg
open Diagrams/Uml/Creational/Prototype.jpg
open Diagrams/Uml/Creational/Singleton.jpg

# Structural Patterns
open Diagrams/Uml/Structural/Adapter.jpg
open Diagrams/Uml/Structural/Bridge.jpg
open Diagrams/Uml/Structural/Composite.jpg
open Diagrams/Uml/Structural/Decorator.jpg
```

## 🔨 Building and Running

### General Build Instructions

Each pattern directory contains its own `Makefile` with the following targets:

```bash
make        # Compile the code
make run    # Execute the compiled program
make clean  # Remove compiled files
```

### Example: Running the Factory Pattern

```bash
# Java Implementation
cd Java/Creational/Factory/
make
make run

# C++ Implementation
cd C++/Creational/Factory/
make
make run

# C# Implementation
cd C#/Creational/Factory/
make
make run
```

### Expected Output Examples

#### Factory Pattern Output
```
Playing the trumpet
Playing the drum
Playing the guitar
```

#### Singleton Pattern Output
```
Consulting Id: [unique-id-value]
```

#### Decorator Pattern Output
```
Basic superhero abilities
Flying power activated!
X-Ray vision activated!
```

## 🎓 Learning Resources

### Books
- **Design Patterns: Elements of Reusable Object-Oriented Software** by Gang of Four
- **Head First Design Patterns** by Eric Freeman and Elisabeth Robson
- **Patterns of Enterprise Application Architecture** by Martin Fowler

### Online Resources
- [Refactoring.Guru - Design Patterns](https://refactoring.guru/design-patterns)
- [SourceMaking - Design Patterns](https://sourcemaking.com/design_patterns)
- [Wikipedia - Software Design Pattern](https://en.wikipedia.org/wiki/Software_design_pattern)

## 🙏 Acknowledgments

- **Gang of Four** for their foundational work on design patterns
