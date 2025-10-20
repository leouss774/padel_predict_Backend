from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token
from flask_cors import CORS
from datetime import timedelta

# Initialisation de l'application
app = Flask(__name__)

# Configuration CORS - TRÈS IMPORTANT
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tdf_predictor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'super-secret-key-2024'
app.config['JWT_SECRET_KEY'] = 'jwt-super-secret-2024'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=1)

# Initialisation des extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Modèle User
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='viewer')
    
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

# Initialisation de la base de données
with app.app_context():
    db.create_all()
    print("✅ Base de données initialisée")

# ==================== ROUTES ====================

# Route de test
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        "message": "✅ Backend TdF Predictor fonctionne parfaitement!",
        "status": "success",
        "version": "1.0"
    })

# Route racine
@app.route('/')
def home():
    return jsonify({
        "message": "🚀 Serveur TdF Predictor démarré avec succès!",
        "endpoints": {
            "test": "/api/test (GET)",
            "create_test_users": "/api/create-test-users (GET)", 
            "login": "/api/auth/login (POST)",
            "register": "/api/auth/register (POST)",
            "debug_users": "/api/debug/users (GET)"
        }
    })

# Route pour créer les utilisateurs de test
@app.route('/api/create-test-users', methods=['GET'])
def create_test_users():
    print("🔄 Création des utilisateurs de test...")
    
    try:
        # Supprimer tous les utilisateurs existants
        User.query.delete()
        db.session.commit()
        print("🗑️ Anciens utilisateurs supprimés")
    except Exception as e:
        print(f"ℹ️ Aucun utilisateur à supprimer: {e}")
        db.session.rollback()
    
    test_users = [
        {'email': 'admin@aso.com', 'password': 'password', 'role': 'admin'},
        {'email': 'team@ineos.com', 'password': 'password', 'role': 'team_manager'},
        {'email': 'sponsor@carrefour.com', 'password': 'password', 'role': 'partner'},
        {'email': 'fan@example.com', 'password': 'password', 'role': 'viewer'},
    ]
    
    created_users = []
    for user_data in test_users:
        # Vérifier si l'utilisateur existe déjà
        existing_user = User.query.filter_by(email=user_data['email']).first()
        if existing_user:
            db.session.delete(existing_user)
            db.session.commit()
        
        new_user = User(email=user_data['email'], role=user_data['role'])
        new_user.set_password(user_data['password'])
        db.session.add(new_user)
        created_users.append(user_data['email'])
        print(f"✅ Utilisateur {user_data['email']} créé")
    
    db.session.commit()
    
    users_count = User.query.count()
    print(f"📊 {users_count} utilisateurs créés avec succès")
    
    return jsonify({
        "message": "Utilisateurs de test créés avec succès",
        "created_users": created_users,
        "total_users": users_count
    })

# Route de connexion
@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        print(f"🔑 Tentative de connexion pour: {data.get('email')}")
        
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({"msg": "Email et mot de passe requis"}), 400
        
        user = User.query.filter_by(email=data['email']).first()
        
        if not user:
            print(f"❌ Utilisateur {data['email']} non trouvé")
            return jsonify({"msg": "Email ou mot de passe incorrect"}), 401
        
        print(f"✅ Utilisateur trouvé: {user.email}")
        
        if user.check_password(data['password']):
            access_token = create_access_token(
                identity=user.id,
                additional_claims={"role": user.role, "email": user.email}
            )
            
            print(f"🎉 Connexion réussie pour {user.email}")
            return jsonify({
                "access_token": access_token,
                "role": user.role,
                "email": user.email
            }), 200
        else:
            print(f"❌ Mot de passe incorrect pour {user.email}")
            return jsonify({"msg": "Email ou mot de passe incorrect"}), 401
            
    except Exception as e:
        print(f"💥 Erreur lors de la connexion: {str(e)}")
        return jsonify({"msg": "Erreur serveur lors de la connexion"}), 500

# Route d'inscription
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        print(f"📝 Tentative d'inscription pour: {data.get('email')}")
        
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({"msg": "Email et mot de passe requis"}), 400
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({"msg": "Un utilisateur avec cet email existe déjà"}), 400
        
        new_user = User(email=data['email'], role='viewer')
        new_user.set_password(data['password'])
        
        db.session.add(new_user)
        db.session.commit()
        
        print(f"✅ Nouvel utilisateur créé: {new_user.email}")
        return jsonify({
            "msg": "Utilisateur créé avec succès",
            "email": new_user.email,
            "role": new_user.role
        }), 201
        
    except Exception as e:
        print(f"💥 Erreur lors de l'inscription: {str(e)}")
        return jsonify({"msg": "Erreur lors de la création du compte"}), 500

@app.route('/api/debug/users', methods=['GET'])
def debug_users():
    users = User.query.all()
    user_list = []
    for user in users:
        user_list.append({
            'id': user.id,
            'email': user.email,
            'role': user.role
        })
    return jsonify({
        "total_users": len(user_list),
        "users": user_list
    })

# Gestion des erreurs 404
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Route non trouvée",
        "message": "La route demandée n'existe pas sur ce serveur",
        "available_routes": [
            "/",
            "/api/test",
            "/api/create-test-users",
            "/api/auth/login", 
            "/api/auth/register",
            "/api/debug/users"
        ]
    }), 404

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 SERVEUR TdF PREDICTOR - DÉMARRAGE")
    print("=" * 60)
    print("📡 URL: http://localhost:8000")
    print("🔗 Test: http://localhost:8000/api/test")
    print("👥 Créer users: http://localhost:8000/api/create-test-users")
    print("🔐 Login: POST http://localhost:8000/api/auth/login")
    print("📝 Register: POST http://localhost:8000/api/auth/register")
    print("🐛 Debug: http://localhost:8000/api/debug/users")
    print("=" * 60)
    
    # Démarrer le serveur
    app.run(host='0.0.0.0', port=8000, debug=True)