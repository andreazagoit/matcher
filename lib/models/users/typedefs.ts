/**
 * GraphQL Schema for Users
 */

export const userTypeDefs = `#graphql
  """User — demographic, auth, and location data."""
  type User {
    id: ID!
    username: String
    name: String!
    email: String!
    birthdate: String!
    image: String
    location: String
    coordinates: Coordinates
    locationUpdatedAt: DateTime
    createdAt: DateTime!
    updatedAt: DateTime!
    userItems: [UserItem!]!
    gender: Gender

    # Orientation & identity
    sexualOrientation: [String!]!
    heightCm: Int

    # Relational intent
    relationshipIntent: [String!]!
    relationshipStyle: RelationshipStyle
    hasChildren: HasChildren
    wantsChildren: WantsChildren

    # Lifestyle
    religion: Religion
    smoking: Smoking
    drinking: Drinking
    activityLevel: ActivityLevel

    # Identity & background
    jobTitle: String
    educationLevel: EducationLevel
    schoolName: String
    languages: [String!]!
    ethnicity: Ethnicity
  }

  input UpdateUserInput {
    username: String
    name: String
    email: String
    birthdate: String
    gender: Gender

    # Orientation & identity
    sexualOrientation: [String!]
    heightCm: Int

    # Relational intent
    relationshipIntent: [String!]
    relationshipStyle: RelationshipStyle
    hasChildren: HasChildren
    wantsChildren: WantsChildren

    # Lifestyle
    religion: Religion
    smoking: Smoking
    drinking: Drinking
    activityLevel: ActivityLevel

    # Identity & background
    jobTitle: String
    educationLevel: EducationLevel
    schoolName: String
    languages: [String!]
    ethnicity: Ethnicity
    location: String
  }

  enum Gender {
    man
    woman
    non_binary
  }

  enum RelationshipStyle {
    monogamous
    ethical_non_monogamous
    open
    other
  }

  enum HasChildren {
    no
    yes
  }

  enum WantsChildren {
    yes
    no
    open
  }

  enum Religion {
    none
    christian
    muslim
    jewish
    buddhist
    hindu
    spiritual
    other
  }

  enum Smoking {
    never
    sometimes
    regularly
  }

  enum Drinking {
    never
    sometimes
    regularly
  }

  enum ActivityLevel {
    sedentary
    light
    moderate
    active
    very_active
  }

  enum EducationLevel {
    middle_school
    high_school
    bachelor
    master
    phd
    vocational
    other
  }

  enum Ethnicity {
    white_caucasian
    hispanic_latino
    black_african
    east_asian
    south_asian
    middle_eastern
    pacific_islander
    indigenous
    mixed
    other
  }

  extend type Query {
    user(username: String!): User
    """Check if a username is already taken."""
    checkUsername(username: String!): Boolean!
    me: User

    """
    AI-recommended events for the authenticated viewer.
    Returns [] for unauthenticated requests.
    """
    recommendedEvents(limit: Int, offset: Int): EventConnection!

    """
    AI-recommended spaces for the authenticated viewer.
    Returns empty connection for unauthenticated requests.
    """
    recommendedSpaces(limit: Int, offset: Int): SpaceConnection!

    """
    Users with similar embeddings to the authenticated viewer.
    Returns [] for unauthenticated requests.
    """
    recommendedUsers(limit: Int, offset: Int): [User!]!

    """
    Categories recommended for the authenticated viewer based on embedding similarity.
    Returns [] for unauthenticated requests.
    """
    recommendedCategories(limit: Int, offset: Int): [Category!]!
  }

  extend type Mutation {
    updateUser(id: ID!, input: UpdateUserInput!): User
    deleteUser(id: ID!): Boolean!
    updateLocation(lat: Float!, lon: Float!, location: String): User!
  }
`;
