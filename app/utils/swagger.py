from flask import jsonify
from app.config.environment import API_VERSION

def generate_swagger_spec():
    """生成Swagger/OpenAPI规范"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Blockchain Research Assistant API",
            "description": "API for analyzing blockchain research papers",
            "version": API_VERSION
        },
        "servers": [
            {
                "url": "/api/v1",
                "description": "Main API server"
            }
        ],
        "paths": {
            "/papers": {
                "post": {
                    "summary": "Upload a paper",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Paper uploaded successfully"
                        }
                    }
                },
                "get": {
                    "summary": "Get recent papers",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {
                                "type": "integer",
                                "default": 10
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of papers"
                        }
                    }
                }
            },
            "/papers/{paper_id}": {
                "get": {
                    "summary": "Get paper details",
                    "parameters": [
                        {
                            "name": "paper_id",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "integer"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Paper details"
                        }
                    }
                }
            },
            "/papers/{paper_id}/analysis": {
                "get": {
                    "summary": "Get paper analysis",
                    "parameters": [
                        {
                            "name": "paper_id",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "integer"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Paper analysis results"
                        }
                    }
                }
            },
            "/qa/ask": {
                "post": {
                    "summary": "Ask a question",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "question": {
                                            "type": "string"
                                        },
                                        "paper_id": {
                                            "type": "integer"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Answer to the question"
                        }
                    }
                }
            }
        }
    }

def setup_swagger(app):
    """设置Swagger文档"""
    @app.route('/api/docs/spec')
    def get_swagger_spec():
        return jsonify(generate_swagger_spec()) 