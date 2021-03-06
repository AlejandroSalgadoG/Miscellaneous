var express = require('express');
var bodyParser = require('body-parser');
var cookieParser = require('cookie-parser');
var fileUpload = require('express-fileupload');

var routes = require('./controller');
var model = require('./model/model');

var app = express();

app.use(cookieParser());
app.use(fileUpload());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.use(express.static(__dirname + '/public'));
app.set('view engine', 'ejs');

// Routes
app.get('/', routes.home);
app.get('/logout', routes.logout);
app.get('/read_users', routes.read_users);
app.get('/manage_account', routes.manage_account);
app.get('/search_images_by_name', routes.search_images_by_name);
app.get('/search_images_by_type', routes.search_images_by_type);

app.post('/login', routes.login);
app.post('/register', routes.register);
app.post('/delete_user', routes.delete_user);
app.post('/update_password', routes.update_password);
app.post('/create_image', routes.create_image);
app.post('/update_image', routes.update_image);
app.post('/share_image', routes.share_image);
app.post('/delete_image', routes.delete_image);

var server = app.listen(3000);
function cleanup(){ server.close(); }

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
