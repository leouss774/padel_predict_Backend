from functools import wraps
from flask import request, jsonify
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request
from models import User, db  # 

def role_required(required_role):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                verify_jwt_in_request()
                current_user_id = get_jwt_identity()
                current_user = User.query.get(current_user_id)
                
                if not current_user:
                    return jsonify({"msg": "Utilisateur non trouvé"}), 404
                    
                if current_user.role != required_role and current_user.role != 'admin':
                    return jsonify({"msg": "Accès interdit: permissions insuffisantes"}), 403
                    
            except Exception as e:
                return jsonify({"msg": "Token invalide ou manquant"}), 401
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator