# image_manager

Aplicación web que permite gestionar imagenes con un simple CRUD de usuarios y archivos.

# Definición de tecnología de desarrollo y despliegue para la aplicación:

* Lenguaje de Programación: Javascript
* Framework web backend: NodeJS - Express
* Framework web frontend: no se usa - se utilizará Templates HTML para Vista (V)
* Base de datos: Mysql
* Web App Server: NodeJS
* Web Server: Apache Web Server

# Estructura de carpetas

* app.js -> (main)
* installation
    - create_database.sql -> (script para la creacion de la base de datos)
    - install_dependencies.sh -> (script para la instalacion de dependencias de node)
* controller
    - index.js
* model
    - model.js
* views
    - account.ejs
    - home.ejs
    - logged.ejs
    - users.ejs
* node_modules -> (carpeta contenedora de las dependencias)
* package.json
* package-lock.json
* README.md

# Dependencias

* express
* body-parser
* cookie-parser
* mysql
* ejs

# Modelo de datos:

|user| 1---* |associations| *---1 |image|

* user
    - user: String -> (Primary key)
    - password: String

* associations
    - userid: String -> (Foreign key to user.user)
    - imageid: String -> (Foreign key to image.name)
    - owner: Bool

* image
    - name: String -> (Primary key)
    - type: String
    - size: int
    - dimension: String
    - private: Bool

# Despliegue en un Servidor Centos 7.x y digital ocean

## Instalacion de nodejs en el servidor

    $ sudo yum install wget
    $ wget http://springdale.math.ias.edu/data/puias/unsupported/7/x86_64//http-parser-2.7.1-3.sdl7.x86_64.rpm
    $ sudo yum install http-parser-2.7.1-3.sdl7.x86_64.rpm
    $ sudo yum install epel-release  # repositorio extra donde se encuentra el paquete de node
    $ sudo yum install nodejs

## Configuracion de hostnames

    Realizar la siguiente configuracion en todos los nodos
    $ sudo hostnamectl set-hostname <hostname>
    $ sudo vim /etc/hosts

    <ip_nodo1>  <hostname_nodo1>
    <ip_nodo2>  <hostname_nodo2>
    <ip_nodo3>  <hostname_nodo3>
    <ip_nodo4>  <hostname_nodo4>
    <ip_nodo5>  <hostname_nodo5>

## Configuracion de Corosync y pacemaker

    $ sudo yum install pacemaker # corosync es instalado como dependencia
    $ sudo yum install haveged # paquete para la creacion de la llave
    $ sudo corosync-keygen # generar la llave para permitir el acceso
    $ sudo cp authkey /etc/corosync # copiar la llave en los otros nodos
    $ sudo vim /etc/corosync/corosync.conf

    totem {
      version: 2
      cluster_name: <nombre cluster>
      transport: udpu
      interface {
        ringnumber: 0
        bindnetaddr: <ip red>
        broadcast: yes
        mcastport: 5405
      }
    }
    
    quorum {
      provider: corosync_votequorum
      two_node: 1
    }
    
    nodelist {
      node {
        ring0_addr: <hostname nodo1>
        name: <nombre nodo1>
        nodeid: 1
      }
      node {
        ring0_addr: <hostname nodo2>
        name: <nombre nodo2>
        nodeid: 2
      }
    }

    logging {
      to_logfile: yes
      logfile: /var/log/corosync/corosync.log
      to_syslog: yes
      timestamp: on
    }

    $ sudo mkdir /var/log/corosync

Configurar peacemaker

    $ sudo mkdir /etc/corosync/service.d
    $ sudo vim /etc/corosync/service.d/pcmk

    service {
        name: pacemaker
        ver: 1
    }

    $ sudo firewall-cmd --zone=public --add-port=5404/udp --permanent
    $ sudo firewall-cmd --zone=public --add-port=5405/udp --permanent
    $ sudo firewall-cmd --zone=public --add-port=5406/udp --permanent
    $ sudo firewall-cmd --reload

    $ sudo systemctl start corosync
    $ sudo systemctl enable corosync
    $ sudo systemctl start pacemaker
    $ sudo systemctl enable  pacemaker

    $ sudo corosync-cmapctl | grep member # Verificar que todos los nodos esten respondiendo

    $ sudo firewall-cmd --zone=public --add-service=high-availability --permanent
    $ sudo firewall-cmd --reload

    $ sudo yum install pcs
    $ sudo systemctl start pcsd
    $ sudo systemctl enable pcsd

    $ sudo passwd hacluster

Los siguientes comandos solo se necesitan ejecutar en un solo nodo

    $ sudo pcs cluster auth <hostname nodo1> <hostname nodo2>
    $ sudo pcs cluster start --all
    $ sudo pcs property set stonith-enabled=false
    $ sudo pcs property set no-quorum-policy=ignore

    $ sudo pcs resource create virtual_ip_haproxy ocf:heartbeat:IPaddr2 ip=<ip virtual> cidr_netmask=24 op monitor interval=10s

## Configuracion HAproxy

    $ sudo yum install haproxy
    $ sudo firewall-cmd --zone=public --add-service=http --permanent
    $ sudo firewall-cmd --reload
    $ sudo setsebool -P haproxy_connect_any 1
    $ sudo vim /etc/haproxy/haproxy.cfg

    frontend http
        bind    <ip flotante>:80
        default_backend app_image
    
    backend app_image
        server  app1 <ip nodo1>:3000 check
        server  app2 <ip nodo2>:3000 check

    listen mysql-cluster
        bind 127.0.0.1:3306
        mode tcp
        option mysql-check user haproxy_check
        balance roundrobin
        server mysql-1 <ip servidor bd 1>:3306 check
        server mysql-2 <ip servidor bd 2>:3306 check

    $ sudo vim /etc/sysctl.conf

    net.ipv4.ip_nonlocal_bind=1

    $ wget https://raw.githubusercontent.com/thisismitch/cluster-agents/master/haproxy
    $ chmod +x haproxy
    $ sudo mv haproxy /usr/lib/ocf/resource.d/heartbeat

Los siguientes comandos solo deben ser ejecutados en un nodo

    $ sudo pcs resource create haproxy ocf:heartbeat:haproxy op monitor interval=10s
    $ sudo pcs resource clone haproxy

## Instalacion de la base de datos

    $ sudo yum install mariadb-server
    $ sudo systemctl start mariadb  # iniciar el servicio de base de datos
    $ sudo systemctl enable mariadb  # configurar el servicio para ejecutarce al iniciar la maquina
    $ sudo /usr/bin/mysql_secure_installation  # configuracion final de la base de datos
    $ sudo firewall-cmd --zone=public --add-service=mysql --permanent
    $ sudo firewall-cmd --reload

    $ mysql -u root -p
    mysql> CREATE DATABASE image_manager;
    mysql> CREATE USER '<user>' IDENTIFIED BY '<password>';
    mysql> GRANT ALL ON image_manager.* TO '<user>' IDENTIFIED BY '<password>';

## Configuracion Master-Master para base de datos

    Servidor 1
    $ sudo vim /etc/my.cnf

    server-id=<id>
    log-bin=mysql-bin

    $ sudo systemctl restart mariadb
    $ mysql -u root -p

    mysql> CREATE USER 'haproxy_check';
    mysql> create user '<reply user>'@'%' identified by '<reply password>';
    mysql> grant replication slave on *.* to '<reply user>'@'%';
    mysql> show master status; # guardar file y position

    Servidor 2
    Repetir pasos anteriores, no olvidar cambiar id en /etc/my.cnf

    mysql> slave stop;
    mysql> change master to master_host='<ip_servidor1>', master_user='<reply user>', master_password='<reply password>', master_log_file='<log_file>', master_log_pos=<log_pos>;
    mysql> slave start;
    mysql> show slave status\G; # No deben salir errores

    Servidor 1

    mysql> slave stop;
    mysql> change master to master_host='<ip_servidor2>', master_user='<user>', master_password='<password>', master_log_file='<log_file>', master_log_pos=<log_pos>;
    mysql> slave start;
    mysql> show slave status\G; # No deben salir errores

## Configuracion de NFS
    Servidor:
    $ sudo yum install nfs-utils
    $ sudo mkdir <carpeta_a_compartir>
    $ sudo chown <usuario>.<grupo> <carpeta_a_compartir>
    $ sudo systemctl start rpcbind
    $ sudo systemctl enable rpcbind
    $ sudo systemctl start nfs-server
    $ sudo systemctl enable nfs-server
    $ sudo vim /etc/exports

    /share <ip_red_clientes>/<mascara>(rw,sync,no_root_squash)

    $ sudo exportfs -r
    $ sudo firewall-cmd --zone=public --add-service=mountd --permanent
    $ sudo firewall-cmd --zone=public --add-service=rpc-bind --permanent
    $ sudo firewall-cmd --zone=public --add-service=nfs --permanent
    $ sudo firewall-cmd --reload
    $ showmount -e localhost

    Cliente:

    $ sudo yum install nfs-utils
    $ sudo mkdir <carpeta_compartida>
    $ sudo chown <usuario>.<grupo> <carpeta_a_compartir>
    $ sudo systemctl start rpcbind
    $ sudo systemctl enable rpcbind
    $ sudo mount <ip_servidor>:/<carpeta_a_compartir> <carpeta_compartida> -o bg,soft,timeo=1
    $ sudo vim /etc/fstab

    <ip_servidor>:<carpeta_a_compartir> <carpeta_compartida>    nfs    bg,soft,timeo=1    0 0

Configuracion de sincronizacion de archivos

    Crontab
    $ sudo yum install rsync
    $ sudo vim /etc/crontab

    * * * * * <user> rsync <other_host>:/share/ /sharebk --delete -r
    
    Rc-local
    $ sudo chmod +x /etc/rc.d/rc.local
    $ sudo vim /etc/rc.d/rc.local
    
    su <user> -c "rsync <other_host>:/sharebk/ /share --delete -r"
    
    $ sudo systemctl start rc-local
    $ sudo systemctl enable rc-local

## Configurar la aplicacion

    $ git clone https://github.com/AlejandroSalgadoG/Web.git  # descargar codigo
    $ cd Web/installation
    $ ./install_dependencies.sh
    $ mysql -h 127.0.0.1 -u <user> -p < create_database.sql

    $ sudo firewall-cmd --zone=public --add-port=3000/tcp --permanent
    $ sudo firewall-cmd --reload

## Configuracion de manejador de procesos

    $ sudo npm install pm2 -g
    # generar el servicio que va a ejecutar la aplicacion al encender la maquina
    $ sudo env PATH=$PATH:/usr/bin /usr/lib/node_modules/pm2/bin/pm2 startup systemd -u asalgad2 --hp /home/asalgad2
    $ pm2 start app.js
    $ pm2 save
