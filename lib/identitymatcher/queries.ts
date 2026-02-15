/**
 * GraphQL query/mutation strings for calling Identity Matcher's API.
 *
 * These are sent as plain strings via the `idmGraphQL` client (server-to-server).
 * They use API key auth, so userId is always required.
 */

// ============================================
// QUERIES
// ============================================

export const IDM_FIND_MATCHES = `
  query FindMatches($userId: ID!, $limit: Int, $gender: [String!], $minAge: Int, $maxAge: Int) {
    findMatches(userId: $userId, limit: $limit, gender: $gender, minAge: $minAge, maxAge: $maxAge) {
      user {
        id
        name
        givenName
        familyName
        image
        gender
        birthdate
      }
      similarity
      breakdown {
        psychological
        values
        interests
        behavioral
      }
    }
  }
`;

export const IDM_PROFILE_STATUS = `
  query ProfileStatus($userId: ID!) {
    profileStatus(userId: $userId) {
      hasAssessment
      hasProfile
      assessmentName
      completedAt
    }
  }
`;

export const IDM_ASSESSMENT_QUESTIONS = `
  query AssessmentQuestions {
    assessmentQuestions {
      section
      questions {
        id
        type
        text
        options
        scaleLabels
        template
        placeholder
      }
    }
  }
`;

// ============================================
// MUTATIONS
// ============================================

export const IDM_SUBMIT_ASSESSMENT = `
  mutation SubmitAssessment($userId: ID!, $answers: JSON!) {
    submitAssessment(userId: $userId, answers: $answers) {
      success
      profileComplete
    }
  }
`;
