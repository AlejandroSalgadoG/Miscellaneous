DROP DATABASE size_db;

CREATE DATABASE size_db;
USE size_db;

CREATE TABLE clients(
    id_client int AUTO_INCREMENT,
    name varchar(32) NOT NULL,
    dir varchar(32) NOT NULL,
    nit varchar(32) NOT NULL,
    tel int NOT NULL,
    PRIMARY KEY (id_client)
);

CREATE TABLE bills(
    id_bill int AUTO_INCREMENT,
    id_client int NOT NULL,
    total int NOT NULL,
    PRIMARY KEY (id_bill),
    CONSTRAINT FK_id_client FOREIGN KEY (id_client) REFERENCES clients(id_client) 
);

CREATE table elements(
    id_element int AUTO_INCREMENT,
    id_bill int NOT NULL,
    descr varchar(64) NOT NULL,
    quant int NOT NULL,
    cost int NOT NULL,
    PRIMARY KEY (id_element),
    CONSTRAINT FK_id_bill FOREIGN KEY (id_bill) REFERENCES bills(id_bill) 
);
