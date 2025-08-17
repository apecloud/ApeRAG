# ApeRAG Quota System Design Document

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Data Model](#data-model)
- [API Design](#api-design)
- [Service Layer](#service-layer)
- [Frontend Implementation](#frontend-implementation)
- [Security & Authorization](#security--authorization)
- [Error Handling](#error-handling)
- [Usage Patterns](#usage-patterns)
- [Future Enhancements](#future-enhancements)

## Overview

The ApeRAG quota system is a comprehensive resource management solution designed to control and monitor user resource consumption across the platform. It provides fine-grained control over various resource types including collections, documents, and bots, ensuring fair usage and preventing system abuse.

### Key Features

- **Multi-resource Quota Management**: Support for different quota types (collections, documents, bots)
- **Real-time Usage Tracking**: Automatic tracking of current resource usage
- **System Default Configuration**: Centralized default quota settings for new users
- **Administrative Controls**: Admin-only quota management and user search capabilities
- **Atomic Operations**: Thread-safe quota consumption and release operations
- **Flexible API Design**: RESTful APIs supporting both individual and batch operations

### Supported Quota Types

1. **max_collection_count**: Maximum number of collections a user can create
2. **max_document_count**: Total maximum number of documents across all collections
3. **max_document_count_per_collection**: Maximum documents per individual collection
4. **max_bot_count**: Maximum number of bots a user can create (excluding system default bot)

## Architecture

The quota system follows a layered architecture pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   User Quotas   │  │  Admin Panel    │  │ System Config│ │
│  │     View        │  │     View        │  │     View     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Quota APIs    │  │   Admin APIs    │  │ System APIs  │ │
│  │  (quotas.yaml)  │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              QuotaService                               │ │
│  │  • get_user_quotas()                                   │ │
│  │  • check_and_consume_quota()                           │ │
│  │  • release_quota()                                     │ │
│  │  • recalculate_user_usage()                            │ │
│  │  • update_user_quota()                                 │ │
│  │  • get/update_system_default_quotas()                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   UserQuota     │  │   ConfigModel   │  │   Related    │ │
│  │     Table       │  │     Table       │  │   Tables     │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Data Model

### UserQuota Table

The core table for storing user quota information:

```sql
CREATE TABLE user_quota (
    user VARCHAR(256) NOT NULL,           -- User identifier
    key VARCHAR(256) NOT NULL,            -- Quota type key
    quota_limit INTEGER NOT NULL DEFAULT 0, -- Maximum allowed usage
    current_usage INTEGER NOT NULL DEFAULT 0, -- Current usage count
    gmt_created TIMESTAMP WITH TIME ZONE NOT NULL,
    gmt_updated TIMESTAMP WITH TIME ZONE NOT NULL,
    gmt_deleted TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (user, key)
);
```

**Key Fields:**
- `user`: User identifier (foreign key to user table)
- `key`: Quota type (e.g., 'max_collection_count', 'max_document_count')
- `quota_limit`: Maximum allowed usage for this quota type
- `current_usage`: Real-time tracking of current usage
- Composite primary key ensures one quota record per user per quota type

### ConfigModel Table

Stores system-wide configuration including default quotas:

```sql
CREATE TABLE config (
    key VARCHAR(256) PRIMARY KEY,         -- Configuration key
    value TEXT NOT NULL,                  -- JSON configuration value
    gmt_created TIMESTAMP WITH TIME ZONE NOT NULL,
    gmt_updated TIMESTAMP WITH TIME ZONE NOT NULL,
    gmt_deleted TIMESTAMP WITH TIME ZONE
);
```

**System Default Quotas Configuration:**
```json
{
  "max_collection_count": 10,
  "max_document_count": 1000,
  "max_document_count_per_collection": 100,
  "max_bot_count": 5
}
```

### Related Tables

The quota system integrates with several other tables for usage calculation:

- **Collection**: For counting user collections (status != 'DELETED')
- **Document**: For counting documents across collections
- **Bot**: For counting user bots (excluding system default bot)

## API Design

### RESTful Endpoints

#### 1. Get User Quotas
```http
GET /quotas?user_id={user_id}&search={search_term}
```

**Features:**
- Current user quotas (no parameters)
- Admin-only user search by ID, username, or email
- Support for multiple search results

**Response Types:**
- `UserQuotaInfo`: Single user quota information
- `UserQuotaList`: Multiple users (search results)

#### 2. Update User Quota
```http
PUT /quotas/{user_id}
```

**Features:**
- Admin-only operation
- Supports both single and batch quota updates
- Atomic transaction handling

**Request Body:**
```json
{
  "max_collection_count": 20,
  "max_document_count": 2000,
  "max_bot_count": 10
}
```

#### 3. Recalculate Usage
```http
POST /quotas/{user_id}/recalculate
```

**Features:**
- Admin-only operation
- Recalculates actual usage from database
- Updates current_usage fields atomically

#### 4. System Default Quotas
```http
GET /system/default-quotas
PUT /system/default-quotas
```

**Features:**
- Admin-only operations
- Centralized default quota management
- Applied to new user initialization

### OpenAPI Specification

The API follows OpenAPI 3.0 specification with:
- Comprehensive schema definitions
- Detailed error response mappings
- Parameter validation rules
- Security requirements (admin-only operations)

## Service Layer

### QuotaService Class

The `QuotaService` class provides the core business logic for quota management:

#### Key Methods

**1. Quota Consumption (Thread-Safe)**
```python
async def check_and_consume_quota(
    self, 
    user_id: str, 
    quota_type: str, 
    amount: int = 1, 
    session=None
) -> None:
    """
    Atomically check and consume quota with SELECT FOR UPDATE
    Raises QuotaExceededException if quota would be exceeded
    """
```

**2. Quota Release**
```python
async def release_quota(
    self, 
    user_id: str, 
    quota_type: str, 
    amount: int = 1, 
    session=None
) -> None:
    """
    Release quota (decrease usage) with transaction safety
    """
```

**3. Usage Recalculation**
```python
async def recalculate_user_usage(self, user_id: str) -> Dict[str, int]:
    """
    Recalculate actual usage from related tables:
    - Collections: COUNT(*) WHERE user=? AND status!='DELETED'
    - Documents: COUNT(*) via JOIN with collections
    - Bots: COUNT(*) WHERE user=? AND title!='Default Agent Bot'
    """
```

**4. User Initialization**
```python
async def initialize_user_quotas(self, user_id: str) -> None:
    """
    Initialize default quotas for new users from system config
    """
```

#### Transaction Management

- **Database Operations**: All quota operations use async database transactions
- **Atomic Updates**: SELECT FOR UPDATE prevents race conditions
- **Session Handling**: Supports both standalone and nested transactions

## Frontend Implementation

### React Component Architecture

The frontend quota management is implemented as a comprehensive React component with:

#### Key Features

**1. Multi-Tab Interface (Admin Only)**
- User Quotas Management
- System Default Configuration

**2. User Search & Selection**
- Real-time search by username, email, or user ID
- Multiple result handling with selection interface
- Clear search functionality

**3. Inline Table Editing**
- Edit mode toggle for quota limits
- Real-time validation
- Batch save operations
- Cancel/revert functionality

**4. Progress Visualization**
- Usage rate progress bars
- Color-coded status (normal/warning/critical)
- Percentage calculations

**5. Administrative Actions**
- Quota recalculation
- System default quota management
- User-specific quota updates

#### Component Structure

```typescript
interface QuotaInfo {
  quota_type: string;
  quota_limit: number;
  current_usage: number;
  remaining: number;
}

interface UserQuotaInfo {
  user_id: string;
  username: string;
  email?: string;
  role: string;
  quotas: QuotaInfo[];
}
```

#### State Management

- **Local State**: Component-level state for UI interactions
- **API Integration**: Direct API calls with loading states
- **Error Handling**: User-friendly error messages with internationalization

## Security & Authorization

### Role-Based Access Control

**Regular Users:**
- View own quotas only
- Read-only access
- No administrative functions

**Admin Users:**
- Full quota management capabilities
- User search and selection
- System configuration access
- Quota recalculation permissions

### API Security

- **Authentication**: Required for all quota endpoints
- **Authorization**: Admin role verification for management operations
- **Input Validation**: Comprehensive parameter validation
- **Rate Limiting**: Implicit through quota system itself

## Error Handling

### Exception Hierarchy

```python
class QuotaExceededException(BusinessException):
    """
    Raised when quota consumption would exceed limits
    Maps to appropriate HTTP status codes:
    - Collection quota: 403 Forbidden
    - Document quota: 403 Forbidden  
    - Bot quota: 403 Forbidden
    """
```

### Error Response Format

```json
{
  "error_code": "COLLECTION_QUOTA_EXCEEDED",
  "code": 1103,
  "message": "已达到max_collection_count的配额限制。当前使用量: 10/10",
  "details": {
    "quota_type": "max_collection_count",
    "quota_limit": 10,
    "current_usage": 10,
    "quota_exceeded": true
  }
}
```

### Frontend Error Handling

- **Internationalized Messages**: Multi-language error display
- **User-Friendly Feedback**: Clear action guidance
- **Graceful Degradation**: Fallback UI states

## Usage Patterns

### Resource Creation Flow

```python
# Example: Creating a new collection
async def create_collection(user_id: str, collection_data: dict):
    async with database_transaction() as session:
        # 1. Check and consume quota atomically
        await quota_service.check_and_consume_quota(
            user_id, 
            'max_collection_count', 
            session=session
        )
        
        # 2. Create the resource
        collection = Collection(**collection_data)
        session.add(collection)
        
        # 3. Commit transaction (quota and resource together)
        await session.commit()
```

### Resource Deletion Flow

```python
# Example: Deleting a collection
async def delete_collection(user_id: str, collection_id: str):
    async with database_transaction() as session:
        # 1. Mark resource as deleted
        collection.status = 'DELETED'
        collection.gmt_deleted = utc_now()
        
        # 2. Release quota
        await quota_service.release_quota(
            user_id, 
            'max_collection_count', 
            session=session
        )
        
        # 3. Commit transaction
        await session.commit()
```

### Admin Operations

```python
# Example: Batch quota update
await quota_service.update_user_quota(
    user_id="user123",
    quota_updates={
        "max_collection_count": 20,
        "max_document_count": 2000,
        "max_bot_count": 10
    }
)
```

## Future Enhancements

### Planned Features

1. **Quota History Tracking**
   - Historical quota changes
   - Usage analytics
   - Trend analysis

2. **Dynamic Quota Adjustment**
   - Usage-based automatic adjustments
   - Temporary quota increases
   - Time-based quotas

3. **Advanced Monitoring**
   - Real-time quota alerts
   - Usage prediction
   - Capacity planning

4. **Integration Enhancements**
   - External quota providers
   - Multi-tenant support
   - API rate limiting integration

### Technical Improvements

1. **Performance Optimization**
   - Quota caching strategies
   - Batch operations
   - Database indexing improvements

2. **Scalability**
   - Distributed quota management
   - Microservice architecture
   - Event-driven updates

3. **Observability**
   - Detailed metrics collection
   - Quota operation tracing
   - Performance monitoring

---

*This document provides a comprehensive overview of the ApeRAG quota system design and implementation. For specific implementation details, refer to the source code in the respective modules.*
